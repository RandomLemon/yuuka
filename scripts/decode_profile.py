#!/usr/bin/env python3
"""Reduce a rocprofv3 kernel trace of ``yuuka_gemv_decode_sweep`` to the decode
matvec table of docs/benchmarks/qwen3.5_4b.md.

The sweep runs every shape in three forms (per-row GEMV, batched GEMV, streaming
LM head), and every cell submits one plan operation ``kSubmissionsPerCell``
times, five warmups followed by twenty-five timed submissions. One submission is one
kernel dispatch, so the dispatch order of a kernel trace lines up with the
printed rows: cell ``i`` owns dispatches ``30*i .. 30*i+29``. Aggregating across
cells instead would mix fourteen shapes and three forms.

Two columns disagree on purpose. ``host_ms`` is the sweep's own median over the
ten timed submissions and includes the submission and the wait; it is the
number docs/benchmarks/qwen3.5_4b.md records. ``device_us`` is the mean kernel
duration from the trace and excludes everything around the launch, so the
difference between them is the per-operation launch cost the decode step pays
579 times. GB/s is reported for both.

Counters are summed per dispatch: rocprofv3 writes one row per shader-engine
instance, and their sum is the dispatch total. ``SQ_WAVES`` is the check for
that: it equals ``grid_size_x * grid_size_y / 32`` on this device, and the
reduction fails when it does not. Instruction counts are normalized per weight
element as ``count * 32 / (n_in * n_out)``, that is wave-level instructions
times the lanes they cover, which is the "instructions per weight" figure of
docs/benchmarks/qwen3.5_4b.md.

Only the Python standard library is used; the rocpd database is plain SQLite.
"""

import argparse
import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict

# bench/hip/gemv_decode_sweep.cpp submits five warmups and twenty-five timed
# submissions per cell, one dispatch each. Keep this in sync with
# kSubmissionsPerCell there.
SUBMISSIONS_PER_CELL = 30

# Dispatch names rocprofv3 reports for the three forms. The names are demangled
# templates, so match the kernel base name and exclude the neighbours.
KERNEL_FILTER = re.compile(r"(gemv_[a-z0-9_]*kernel|lm_head_kernel)")
FORM_KERNEL = {
    "gemv": re.compile(r"::gemv_(k|q6_k|q8_0|f32|f16|q4_0|q5_0)_kernel"),
    "gemv_batch4": re.compile(r"::gemv_batch_kernel"),
    "lm_head": re.compile(r"::lm_head_kernel"),
}

# Packed row size per dtype, mirroring row_bytes() in
# bench/hip/gemv_decode_sweep.cpp.
ROW_BYTES = {
    "F16": lambda n_in: n_in * 2,
    "Q4_0": lambda n_in: n_in // 32 * 18,
    "Q5_0": lambda n_in: n_in // 32 * 22,
    "Q8_0": lambda n_in: n_in // 32 * 34,
    "Q4_K": lambda n_in: n_in // 256 * 144,
    "Q5_K": lambda n_in: n_in // 256 * 176,
    "Q6_K": lambda n_in: n_in // 256 * 210,
}


def row_bytes(dtype, n_in):
    """Return the packed bytes of one weight row."""
    if dtype not in ROW_BYTES:
        raise Fail(f"unsupported dtype {dtype}; the sweep and this reduction disagree")
    return ROW_BYTES[dtype](n_in)


WAVES = "SQ_WAVES"
INSTRUCTION_COUNTERS = ("SQ_INSTS_VALU", "SQ_INSTS_LDS")

CELL_COLUMNS = ("name", "dtype", "n_in", "n_out", "ms", "GB/s", "ns_per_row",
                "form", "batch")


class Fail(Exception):
    """A user-visible failure: bad input, unsupported schema, or broken pairing."""


class Database:
    """A rocpd database with lazily resolved, UUID-suffixed table names."""

    def __init__(self, path):
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "**", "*_results.db"),
                                     recursive=True))
            if len(found) != 1:
                raise Fail(f"{path}: expected exactly one rocprofv3 *_results.db,"
                           f" found {len(found)}")
            path = found[0]
        if not os.path.isfile(path):
            raise Fail(f"missing rocprofv3 database: {path}")
        self.path = path
        try:
            self.conn = sqlite3.connect(path)
        except sqlite3.Error as error:
            raise Fail(f"{path}: cannot open database: {error}") from error
        self.tables = [row[0] for row in self.conn.execute(
            "select name from sqlite_master where type='table'")]

    def table(self, base):
        prefix = "rocpd_" + base + "_"
        for name in self.tables:
            if name.startswith(prefix):
                return name
        raise Fail(f"{self.path}: missing rocpd table '{base}' (unsupported schema?)")

    def dispatches(self):
        """Return the matvec dispatches in submission order."""
        kernel_dispatch = self.table("kernel_dispatch")
        kernel_symbol = self.table("info_kernel_symbol")
        query = (
            f"select kd.event_id, kd.start, kd.end, kd.grid_size_x, kd.grid_size_y,"
            f" kd.workgroup_size_x, ks.display_name"
            f" from {kernel_dispatch} kd"
            f" join {kernel_symbol} ks on ks.id = kd.kernel_id"
            f" order by kd.start, kd.id"
        )
        selected = []
        try:
            for row in self.conn.execute(query):
                name = row[6] or ""
                if KERNEL_FILTER.search(name):
                    selected.append({
                        "event_id": row[0],
                        "ns": float(row[2]) - float(row[1]),
                        "grid_x": row[3],
                        "grid_y": row[4],
                        "workgroup_x": row[5],
                        "name": name,
                    })
        except sqlite3.Error as error:
            raise Fail(f"{self.path}: cannot read kernel dispatches: {error}") from error
        if not selected:
            raise Fail(f"{self.path}: no gemv or lm_head dispatch; was the sweep"
                       " run under rocprofv3?")
        return selected

    def counters(self, wanted):
        """Return {event_id: {counter: value}} with instances summed."""
        pmc_event = self.table("pmc_event")
        info_pmc = self.table("info_pmc")
        names = {}
        try:
            for pmc_id, name in self.conn.execute(f"select id, name from {info_pmc}"):
                names[pmc_id] = name
        except sqlite3.Error as error:
            raise Fail(f"{self.path}: cannot read pmc definitions: {error}") from error
        present = set(names.values())
        missing = [name for name in wanted if name not in present]
        if missing:
            raise Fail(
                f"{self.path}: missing counters {', '.join(missing)}; the database was"
                " not collected with the expected --pmc groups")
        values = defaultdict(lambda: defaultdict(float))
        try:
            for event_id, pmc_id, value in self.conn.execute(
                    f"select event_id, pmc_id, value from {pmc_event}"):
                name = names.get(pmc_id)
                if name in wanted:
                    values[event_id][name] += float(value)
        except sqlite3.Error as error:
            raise Fail(f"{self.path}: cannot read pmc values: {error}") from error
        return values


def parse_cells(path):
    """Return the sweep cells in run order."""
    cells = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            header = None
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                fields = line.split(",")
                if header is None:
                    if fields != list(CELL_COLUMNS):
                        raise Fail(f"{path}: header {','.join(fields)} does not match"
                                   f" {','.join(CELL_COLUMNS)}")
                    header = fields
                    continue
                if len(fields) != len(CELL_COLUMNS):
                    raise Fail(f"{path}: row {line!r} has {len(fields)} fields, want"
                               f" {len(CELL_COLUMNS)}")
                cell = dict(zip(CELL_COLUMNS, fields))
                for key in ("n_in", "n_out", "batch"):
                    cell[key] = int(cell[key])
                for key in ("ms", "GB/s", "ns_per_row"):
                    cell[key] = float(cell[key])
                cells.append(cell)
    except OSError as error:
        raise Fail(f"{path}: cannot read cells: {error}") from error
    if not cells:
        raise Fail(f"{path}: no sweep cells; the sweep produced no output")
    return cells


def pair_cells(cells, dispatches):
    """Group the dispatches by cell and validate the pairing."""
    expected = len(cells) * SUBMISSIONS_PER_CELL
    if len(dispatches) != expected:
        raise Fail(f"expected {expected} matvec dispatches"
                   f" ({len(cells)} cells x {SUBMISSIONS_PER_CELL} submissions) but"
                   f" found {len(dispatches)}; the profiled run exercised a different"
                   " number of submissions than the sweep printed")
    groups = []
    for index, cell in enumerate(cells):
        group = dispatches[index * SUBMISSIONS_PER_CELL:
                            (index + 1) * SUBMISSIONS_PER_CELL]
        pattern = FORM_KERNEL[cell["form"]]
        for dispatch in group:
            if not pattern.search(dispatch["name"]):
                raise Fail(f"cell {index} ({cell['form']} {cell['name']}): dispatch"
                           f" {dispatch['name']!r} does not belong to this form; the"
                           " dispatch/cell pairing is broken")
        groups.append(group)
    return groups


def cell_metrics(cell, group, counters):
    """Merge the host timing, the device timing, and the counters of one cell."""
    device_ns = sum(dispatch["ns"] for dispatch in group) / len(group)
    waves = sum(counters.get(dispatch["event_id"], {}).get(WAVES, 0.0)
                for dispatch in group)
    # Counter values are per dispatch in rocprofv3's rocpd output, so a cell's
    # figure is the mean over its submissions, not their sum.
    values = {}
    for name in INSTRUCTION_COUNTERS:
        total = sum(counters.get(dispatch["event_id"], {}).get(name, 0.0)
                    for dispatch in group)
        values[name] = total / len(group)
    weights = cell["n_in"] * cell["n_out"]
    return {
        "device_ms": device_ns / 1e6,
        "waves": waves / len(group),
        "valu_per_weight": (values["SQ_INSTS_VALU"] * 32 / weights
                            if weights else 0.0),
        "lds_per_weight": (values["SQ_INSTS_LDS"] * 32 / weights
                           if weights else 0.0),
    }


def format_value(value, digits=2):
    if value is None:
        return "-"
    if digits == 0:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def print_table(label, columns, rows):
    print(label)
    widths = []
    for index, name in enumerate(columns):
        width = len(name)
        for row in rows:
            width = max(width, len(str(row[index])))
        widths.append(width)
    header = "  ".join(name.ljust(widths[index]) for index, name in enumerate(columns))
    print(header)
    for row in rows:
        print("  ".join(str(value).ljust(widths[index])
                        for index, value in enumerate(row)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reduce a rocprofv3 trace of yuuka_gemv_decode_sweep to the decode"
                    " matvec table of docs/benchmarks/qwen3.5_4b.md.")
    parser.add_argument("--cells", required=True,
                        help="sweep output CSV (build/yuuka_gemv_decode_sweep)")
    parser.add_argument("--db", required=True,
                        help="rocprofv3 rocpd database or its directory")
    parser.add_argument("--waves-only", action="store_true",
                        help="accept a trace without SQ_INSTS_VALU/SQ_INSTS_LDS")
    args = parser.parse_args(argv)

    try:
        cells = parse_cells(args.cells)
        database = Database(args.db)
        dispatches = database.dispatches()
        groups = pair_cells(cells, dispatches)
        for cell in cells:
            cell["weight_bytes"] = row_bytes(cell["dtype"], cell["n_in"]) * cell["n_out"]
        wanted = (WAVES,) if args.waves_only else (WAVES,) + INSTRUCTION_COUNTERS
        counters = database.counters(wanted)
        print(f"cells={len(cells)} dispatches={len(dispatches)}"
              f" submissions_per_cell={SUBMISSIONS_PER_CELL}")
        print(f"database={database.path}")
        # SQ_WAVES is the pairing check: one wave per 32 work items, so the
        # per-instance counter rows must sum to the grid size.
        bad = []
        for cell, group in zip(cells, groups):
            dispatch = group[0]
            expected_waves = dispatch["grid_x"] * max(dispatch["grid_y"], 1) / 32
            got = counters.get(dispatch["event_id"], {}).get(WAVES, 0.0)
            if expected_waves and abs(got - expected_waves) > 1:
                bad.append(f"{cell['name']}/{cell['form']}: SQ_WAVES {got:.0f}"
                           f" against {expected_waves:.0f} from the grid")
        if bad:
            raise Fail("SQ_WAVES does not match the launch geometry, so the"
                       " per-instance counter rows are not being summed as the"
                       " dispatch total: " + "; ".join(bad[:3]))
        print()

        rows = []
        by_form = defaultdict(lambda: {"cells": 0, "host_ms": 0.0, "device_ms": 0.0,
                                       "bytes": 0.0})
        for cell, group in zip(cells, groups):
            metrics = cell_metrics(cell, group, counters)
            gbps_device = (cell["weight_bytes"] / (metrics["device_ms"] * 1e6)
                           if metrics["device_ms"] else 0.0)
            rows.append([
                cell["form"], cell["name"], cell["dtype"],
                cell["n_in"], cell["n_out"], cell["batch"],
                format_value(cell["ms"], 4), format_value(metrics["device_ms"], 4),
                format_value(cell["GB/s"], 1), format_value(gbps_device, 1),
                format_value(metrics["waves"], 0) if not args.waves_only else "-",
                format_value(metrics["valu_per_weight"], 1) if not args.waves_only else "-",
                format_value(metrics["lds_per_weight"], 2) if not args.waves_only else "-",
            ])
            entry = by_form[cell["form"]]
            entry["cells"] += 1
            entry["host_ms"] += cell["ms"]
            entry["device_ms"] += metrics["device_ms"]
            entry["bytes"] += cell["weight_bytes"]
        print_table(
            "per cell (host_ms is the sweep median, device_us the traced kernel):",
            ["form", "name", "dtype", "n_in", "n_out", "batch", "host_ms",
             "device_ms", "GB/s_host", "GB/s_device", "waves", "valu/w", "lds/w"],
            rows)
        print()

        summary = []
        for form in sorted(by_form):
            entry = by_form[form]
            summary.append([
                form, entry["cells"],
                format_value(entry["host_ms"], 3),
                format_value(entry["device_ms"], 3),
                format_value(entry["bytes"] / (entry["host_ms"] * 1e6), 1),
                format_value(entry["bytes"] / (entry["device_ms"] * 1e6), 1),
            ])
        print_table("by form (all cells summed):",
                    ["form", "cells", "host_ms", "device_ms", "GB/s_host",
                     "GB/s_device"], summary)
        print()
        print("Interpretation: host_ms - device_ms is the per-submission launch cost"
              " the decode step pays once per operation. valu/w and lds/w are the"
              " wave-level instructions times 32 lanes per packed weight, the figure"
              " docs/benchmarks/qwen3.5_4b.md uses; a gemv_batch4 cell covers four rows,"
              " so its instruction figures are the cost of producing four rows at"
              " once, not four times the per-row cost.")
    except Fail as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
