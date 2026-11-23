#!/usr/bin/env python3
"""Reduce rocprofv3 databases to the per-MMA prefill table of docs/benchmarks/qwen3.5_4b.md.

The reduction pairs ``gemm_mmq`` dispatches with ``yuuka_prefill_sweep`` cells in
submission order. The sweep submits every cell seven times (two warmups plus
five timed runs), and every K-quant submission launches one
``quantize_q8_1_kernel`` and one ``gemm_mmq_kernel``, so cell ``i`` owns
dispatches ``7*i .. 7*i+6``. Aggregating across cells instead would mix nine
different shapes and blur the per-MMA denominator.

Denominator, recorded here because every later stage is judged on it:

* a gfx1151 int8 WMMA is 16 x 16 x 16, that is 4096 multiply-accumulate values;
* ``n_in * n_out * rows / 4096`` is the useful MMA count of a cell, the
  denominator the reduction uses;
* the executed MMA count is
  ``blocks_x * blocks_y * (n_in / 16) * 32``, because ``gemm_mmq_kernel`` runs
  ``kTileI = 64`` output channels by ``kTileJ = 128`` tokens with eight waves
  that each issue four 16x16x16 MMAs per 16-value K slice. Both counts agree
  for cells that fill the tile. Sweep cells whose ``rows`` or ``n_out`` leave
  part of the tile unused execute the padded tile anyway, so this tool
  normalizes every cell by the executed count and marks such cells as padded.

Instruction counters are summed over the seven dispatches of a cell. Percentage
and cycle counters (``LDSBankConflict``, ``ALUStalledByLDS``, ``LdsLatency``,
``OccupancyPercent``) are already aggregated per dispatch by rocprofv3 and are
averaged over the same seven dispatches.

Only the Python standard library is used. The ``rocpd`` Python package is not
required because the database is plain SQLite. Metric values are reported as
rocprofv3 derives them; ``OccupancyPercent`` in particular is not clamped to 100
and reaches about 150% on some 256-thread Q6_K cells on gfx1151, so compare the
occupancy column across runs, not against a physical ceiling.
"""

import argparse
import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict

# The sweep in bench/hip/prefill_sweep.cpp calls bench_rows() with five timed
# iterations after two warmups.
SWEEP_SUBMISSIONS_PER_CELL = 7

# Tile geometry of gemm_mmq_kernel in kernels/hip/gemm/mmq_gemm.hip. If the
# kernel changes kTileI or kTileJ this must change with it; the launch-geometry
# check below fails loudly rather than mis-normalizing.
MMQ_TILE_OUT = 64
MMQ_TILE_ROWS = 128
MMQ_MMAS_PER_BLOCK_SLICE = 32  # 8 waves x 4 accumulator fragments
MMA_VALUES = 4096  # 16 x 16 x 16 int8 WMMA

INSTRUCTION_COUNTERS = ("SQ_INSTS_VALU", "SQ_INSTS_LDS", "SQ_WAVES")
STALL_COUNTERS = (
    "LDSBankConflict",
    "ALUStalledByLDS",
    "LdsLatency",
    "OccupancyPercent",
)

FOCUS = {"dtype": "Q4_K", "rows": 512, "n_out": 9216, "n_in": 2560}

# Recorded 2026-11-18 attribution from docs/benchmarks/qwen3.5_4b.md. These are
# printed as a comparison column only; they are not gates.
RECORDED_YUUKA = {
    "valu_per_mma": 24.8,
    "lds_per_mma": 4.9,
    "lds_bank_conflict_pct": 35.9,
    "alu_stalled_by_lds_pct": 0.3,
    "lds_latency_cycles": 67.6,
    "occupancy_pct": 23.9,
    "waves_per_workgroup": 8.0,
    "achieved_tf_s": 12.6,
}
RECORDED_LLAMA = {
    "valu_per_mma": 17.1,
    "lds_per_mma": 2.0,
    "lds_bank_conflict_pct": 11.7,
    "alu_stalled_by_lds_pct": 4.3,
    "lds_latency_cycles": 127.3,
    "occupancy_pct": 16.7,
    "waves_per_workgroup": 4.0,
    "achieved_tf_s": 21.9,
}

# Order and labels of the printed comparison table.
METRICS = (
    ("valu_per_mma", "VALU instructions per MMA"),
    ("lds_per_mma", "LDS instructions per MMA"),
    ("lds_bank_conflict_pct", "LDSBankConflict (%)"),
    ("alu_stalled_by_lds_pct", "ALU stalled by LDS (%)"),
    ("lds_latency_cycles", "LdsLatency (cycles)"),
    ("occupancy_pct", "OccupancyPercent (%)"),
    ("waves_per_workgroup", "Waves per workgroup"),
    ("achieved_tf_s", "Achieved TF/s"),
)

CSV_COLUMNS = (
    "engine",
    "dtype",
    "rows",
    "n_out",
    "n_in",
    "kernel",
    "dispatches",
    "mmas_useful",
    "mmas_executed",
    "valu_per_mma",
    "lds_per_mma",
    "lds_bank_conflict_pct",
    "alu_stalled_by_lds_pct",
    "lds_latency_cycles",
    "occupancy_pct",
    "waves_per_workgroup",
    "achieved_tf_s",
)

CELL_RE = re.compile(
    r"^(?P<dtype>[A-Za-z0-9_]+) +M= *(?P<rows>\d+) +N= *(?P<n_out>\d+) +K= *(?P<n_in>\d+)"
)
MMQ_TF_RE = re.compile(r"\bmmq=[0-9.]+ms [0-9.]+GB/s +(?P<tf>[0-9.]+)TF/s")


class Fail(Exception):
    """A user-visible failure: bad input, unsupported schema, or broken pairing."""


class Dispatch:
    """One kernel dispatch record with the launch geometry the reduction needs."""

    __slots__ = (
        "id",
        "event_id",
        "name",
        "grid_x",
        "grid_y",
        "workgroup_x",
        "workgroup_y",
    )

    def __init__(self, row):
        (
            self.id,
            self.event_id,
            self.name,
            self.grid_x,
            self.grid_y,
            self.workgroup_x,
            self.workgroup_y,
        ) = row

    @property
    def blocks(self):
        if self.workgroup_x <= 0 or self.workgroup_y <= 0:
            raise Fail(f"dispatch {self.id}: invalid workgroup size")
        if self.grid_x % self.workgroup_x or self.grid_y % self.workgroup_y:
            raise Fail(
                f"dispatch {self.id}: grid {self.grid_x}x{self.grid_y} is not a "
                f"multiple of workgroup {self.workgroup_x}x{self.workgroup_y}"
            )
        return self.grid_x // self.workgroup_x, self.grid_y // self.workgroup_y


class Database:
    """A rocpd database with lazily resolved, UUID-suffixed table names."""

    def __init__(self, path):
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "**", "*_results.db"), recursive=True))
            if len(found) != 1:
                raise Fail(
                    f"{path}: expected exactly one rocprofv3 *_results.db, found {len(found)}"
                )
            path = found[0]
        if not os.path.isfile(path):
            raise Fail(f"missing rocprofv3 database: {path}")
        self.path = path
        try:
            self.conn = sqlite3.connect(path)
        except sqlite3.Error as error:
            raise Fail(f"{path}: cannot open database: {error}") from error
        self.tables = [
            row[0]
            for row in self.conn.execute("select name from sqlite_master where type='table'")
        ]

    def table(self, base):
        prefix = "rocpd_" + base + "_"
        for name in self.tables:
            if name.startswith(prefix):
                return name
        raise Fail(f"{self.path}: missing rocpd table '{base}' (unsupported schema?)")

    def schema_version(self):
        try:
            table = self.table("metadata")
            for tag, value in self.conn.execute(f"select tag, value from {table}"):
                if tag == "schema_version":
                    return value
        except sqlite3.Error:
            pass
        return "unknown"

    def dispatches(self, pattern):
        kernel_dispatch = self.table("kernel_dispatch")
        kernel_symbol = self.table("info_kernel_symbol")
        query = (
            f"select kd.id, kd.event_id, ks.display_name, kd.grid_size_x,"
            f" kd.grid_size_y, kd.workgroup_size_x, kd.workgroup_size_y"
            f" from {kernel_dispatch} kd"
            f" join {kernel_symbol} ks on ks.id = kd.kernel_id"
            f" order by kd.start, kd.id"
        )
        selected = []
        try:
            for row in self.conn.execute(query):
                if pattern.search(row[2] or ""):
                    selected.append(Dispatch(row))
        except sqlite3.Error as error:
            raise Fail(f"{self.path}: cannot read kernel dispatches: {error}") from error
        return selected

    def counters(self, wanted):
        pmc_event = self.table("pmc_event")
        info_pmc = self.table("info_pmc")
        names = {}
        try:
            for pmc_id, name in self.conn.execute(f"select id, name from {info_pmc}"):
                names[pmc_id] = name
        except sqlite3.Error as error:
            raise Fail(f"{self.path}: cannot read pmc definitions: {error}") from error
        missing = [name for name in wanted if name not in set(names.values())]
        if missing:
            raise Fail(
                f"{self.path}: missing counters {', '.join(missing)}; "
                "the database was not collected with the expected --pmc groups"
            )
        values = defaultdict(lambda: defaultdict(float))
        try:
            for event_id, pmc_id, value in self.conn.execute(
                f"select event_id, pmc_id, value from {pmc_event}"
            ):
                name = names.get(pmc_id)
                if name in wanted:
                    values[event_id][name] += float(value)
        except sqlite3.Error as error:
            raise Fail(f"{self.path}: cannot read pmc values: {error}") from error
        return values


def parse_cells(path):
    """Return the K-quant sweep cells that ran the `mmq` route, in run order."""
    cells = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if "mmq=" not in line:
                    continue
                match = CELL_RE.match(line.strip())
                if not match:
                    raise Fail(f"{path}: unparsable sweep line: {line.strip()}")
                tf = MMQ_TF_RE.search(line)
                cells.append(
                    {
                        "dtype": match.group("dtype"),
                        "rows": int(match.group("rows")),
                        "n_out": int(match.group("n_out")),
                        "n_in": int(match.group("n_in")),
                        "achieved_tf_s": float(tf.group("tf")) if tf else None,
                    }
                )
    except OSError as error:
        raise Fail(f"cannot read sweep log {path}: {error}") from error
    if not cells:
        raise Fail(f"{path}: no cells with an 'mmq=' column")
    return cells


def check_launch(cell, index, dispatch, submission):
    """Fail when the paired dispatch does not belong to the expected cell."""
    blocks_x, blocks_y = dispatch.blocks
    if dispatch.workgroup_x != 256:
        raise Fail(
            f"cell {index} submission {submission}: unexpected workgroup size "
            f"{dispatch.workgroup_x} for gemm_mmq_kernel (expected 256)"
        )
    expected_x = -(-cell["n_out"] // MMQ_TILE_OUT)
    expected_y = -(-cell["rows"] // MMQ_TILE_ROWS)
    if blocks_x != expected_x or blocks_y != expected_y:
        raise Fail(
            f"cell {index} submission {submission}: launch covers "
            f"{blocks_y * MMQ_TILE_ROWS} rows x {blocks_x * MMQ_TILE_OUT} "
            f"output channels but the sweep cell is rows={cell['rows']} "
            f"n_out={cell['n_out']}; the dispatch/cell pairing is broken "
            "(tile geometry or sweep order changed)"
        )
    return blocks_x * blocks_y


def pair_cells(cells, dispatches):
    expected = len(cells) * SWEEP_SUBMISSIONS_PER_CELL
    if len(dispatches) != expected:
        raise Fail(
            f"expected {expected} gemm_mmq dispatches "
            f"({len(cells)} cells x {SWEEP_SUBMISSIONS_PER_CELL} submissions), "
            f"found {len(dispatches)}; the profiled run did not exercise the "
            "same cells as the reference sweep"
        )
    paired = []
    for index, cell in enumerate(cells):
        group = dispatches[
            index * SWEEP_SUBMISSIONS_PER_CELL : (index + 1) * SWEEP_SUBMISSIONS_PER_CELL
        ]
        blocks = [
            check_launch(cell, index, dispatch, submission)
            for submission, dispatch in enumerate(group)
        ]
        paired.append((cell, group, sum(blocks)))
    return paired


def cell_metrics(cell, group, blocks, instructions, stalls):
    n_in = cell["n_in"]
    if n_in % 16:
        raise Fail(f"{cell['dtype']} n_in={n_in}: K is not a multiple of 16")
    # `blocks` is the per-cell total of launched workgroups across the seven
    # dispatches; every workgroup issues one MMA per 16-value K slice per
    # accumulator fragment group.
    executed = blocks * (n_in // 16) * MMQ_MMAS_PER_BLOCK_SLICE
    useful_dispatch = n_in * cell["n_out"] * cell["rows"]
    if useful_dispatch % MMA_VALUES:
        raise Fail(
            f"{cell['dtype']} rows={cell['rows']} n_out={cell['n_out']} "
            f"n_in={n_in}: useful MMA count is not integral"
        )
    useful = len(group) * (useful_dispatch // MMA_VALUES)

    sums = defaultdict(float)
    for dispatch in group:
        for name, value in instructions.get(dispatch.event_id, {}).items():
            sums[name] += value
        for name, value in stalls.get(dispatch.event_id, {}).items():
            sums[name + ".stall"] += value
    for name in INSTRUCTION_COUNTERS:
        if name not in sums:
            raise Fail(f"{cell['dtype']} rows={cell['rows']}: no {name} counter")
    for name in STALL_COUNTERS:
        if name + ".stall" not in sums:
            raise Fail(f"{cell['dtype']} rows={cell['rows']}: no {name} counter")

    if executed == 0:
        raise Fail(f"{cell['dtype']} rows={cell['rows']}: zero executed MMAs")
    waves = sums["SQ_WAVES"]
    metrics = {
        "dispatches": len(group),
        "mmas_executed": executed,
        "mmas_useful": useful,
        "valu_per_mma": sums["SQ_INSTS_VALU"] / executed,
        "lds_per_mma": sums["SQ_INSTS_LDS"] / executed,
        "waves_per_workgroup": waves / blocks if blocks else None,
        "achieved_tf_s": cell["achieved_tf_s"],
    }
    key = {
        "LDSBankConflict": "lds_bank_conflict_pct",
        "ALUStalledByLDS": "alu_stalled_by_lds_pct",
        "LdsLatency": "lds_latency_cycles",
        "OccupancyPercent": "occupancy_pct",
    }
    for name, column in key.items():
        metrics[column] = sums[name + ".stall"] / len(group)
    metrics["padded"] = executed != useful
    return metrics


def reference_rows(instructions_path, stalls_path, pattern, mma_counts_path):
    """Aggregate the matched reference kernels, optionally normalized per MMA.

    The reference is traced twice (instruction counters, then stall counters),
    so its dispatches are paired by position inside each database. Both runs
    execute the same kernel sequence; a count or name mismatch fails loudly.
    """
    instructions_db = Database(instructions_path)
    instructions_dispatches = instructions_db.dispatches(pattern)
    if not instructions_dispatches:
        raise Fail(
            f"{instructions_path}: no kernel dispatch matches {pattern.pattern!r}; "
            "pass --reference-kernel-regex for this llama.cpp build"
        )
    stalls_db = Database(stalls_path)
    stalls_dispatches = stalls_db.dispatches(pattern)
    if len(stalls_dispatches) != len(instructions_dispatches):
        raise Fail(
            f"reference runs disagree: {len(instructions_dispatches)} instruction "
            f"dispatches against {len(stalls_dispatches)} stall dispatches"
        )
    for position, (first, second) in enumerate(
        zip(instructions_dispatches, stalls_dispatches)
    ):
        if first.name != second.name:
            raise Fail(
                f"reference run {position}: {first.name!r} in the instruction "
                f"database but {second.name!r} in the stall database"
            )

    print(f"reference matched dispatches: {len(instructions_dispatches)}")
    for name in sorted({dispatch.name for dispatch in instructions_dispatches}):
        count = sum(1 for dispatch in instructions_dispatches if dispatch.name == name)
        print(f"  {count:6d} x {name}")

    instructions = instructions_db.counters(set(INSTRUCTION_COUNTERS))
    stalls = stalls_db.counters(set(STALL_COUNTERS))

    mma_counts = None
    if mma_counts_path:
        mma_counts = [int(line, 0) for line in read_int_list(mma_counts_path)]
        if len(mma_counts) != len(instructions_dispatches):
            raise Fail(
                f"{mma_counts_path}: {len(mma_counts)} MMA counts for "
                f"{len(instructions_dispatches)} matched dispatches"
            )

    rows = []
    for name in sorted({dispatch.name for dispatch in instructions_dispatches}):
        positions = [
            position
            for position, dispatch in enumerate(instructions_dispatches)
            if dispatch.name == name
        ]
        rows.append(
            aggregate_reference(
                name, positions, instructions_dispatches, stalls_dispatches,
                instructions, stalls, mma_counts,
            )
        )
    return rows


def read_int_list(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if line:
                    yield line
    except OSError as error:
        raise Fail(f"cannot read {path}: {error}") from error


def aggregate_reference(name, positions, instructions_dispatches, stalls_dispatches,
                        instructions, stalls, mma_counts):
    sums = defaultdict(float)
    for position in positions:
        for counter, value in instructions.get(
            instructions_dispatches[position].event_id, {}
        ).items():
            sums[counter] += value
        for counter, value in stalls.get(stalls_dispatches[position].event_id, {}).items():
            sums[counter + ".stall"] += value
    for counter in INSTRUCTION_COUNTERS:
        if counter not in sums:
            raise Fail(f"{name}: no {counter} counter")
    for counter in STALL_COUNTERS:
        if counter + ".stall" not in sums:
            raise Fail(f"{name}: no {counter} counter")
    waves = sums["SQ_WAVES"]
    workgroups = 0
    for position in positions:
        blocks_x, blocks_y = instructions_dispatches[position].blocks
        workgroups += blocks_x * blocks_y
    metrics = {
        "dispatches": len(positions),
        "waves": waves,
        "workgroups": workgroups,
        "waves_per_workgroup": waves / workgroups if workgroups else None,
        "valu_per_wave": sums["SQ_INSTS_VALU"] / waves if waves else None,
        "lds_per_wave": sums["SQ_INSTS_LDS"] / waves if waves else None,
    }
    for counter, column in (
        ("LDSBankConflict", "lds_bank_conflict_pct"),
        ("ALUStalledByLDS", "alu_stalled_by_lds_pct"),
        ("LdsLatency", "lds_latency_cycles"),
        ("OccupancyPercent", "occupancy_pct"),
    ):
        metrics[column] = sums[counter + ".stall"] / len(positions)
    if mma_counts is not None:
        total = sum(mma_counts[position] for position in positions)
        if total <= 0:
            raise Fail(f"--reference-mma-counts: zero total MMA count for {name}")
        metrics["mmas_executed"] = total
        metrics["valu_per_mma"] = sums["SQ_INSTS_VALU"] / total
        metrics["lds_per_mma"] = sums["SQ_INSTS_LDS"] / total
    else:
        metrics["mmas_executed"] = None
        metrics["valu_per_mma"] = None
        metrics["lds_per_mma"] = None
    return {"kernel": name, "metrics": metrics}


def focus_cell(cells, paired):
    for cell, group, blocks in paired:
        if all(cell[key] == value for key, value in FOCUS.items()):
            return cell, group, blocks
    raise Fail(
        "the sweep log has no Q4_K M=512 N=9216 K=2560 cell; "
        "the recorded attribution cannot be reproduced from this run"
    )


def format_value(value, digits=2):
    if value is None:
        return "na"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def print_table(label, rows, columns):
    widths = [len(column) for column in columns]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print(f"\n{label}")
    print("  ".join(column.ljust(width) for column, width in zip(columns, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reduce rocprofv3 databases to per-MMA prefill counters.",
    )
    parser.add_argument("--cells", required=True, help="yuuka_prefill_sweep stdout log")
    parser.add_argument(
        "--instructions-db", required=True, help="rocpd database with SQ_INSTS_* counters"
    )
    parser.add_argument(
        "--stalls-db", required=True, help="rocpd database with LDS/occupancy counters"
    )
    parser.add_argument(
        "--reference-instructions-db", help="rocpd database around the llama-bench run"
    )
    parser.add_argument(
        "--reference-stalls-db",
        help="second rocpd database around the llama-bench run with stall counters",
    )
    parser.add_argument(
        "--reference-kernel-regex",
        default="mul_mat_q",
        help="regex selecting reference MMQ kernels (default: mul_mat_q)",
    )
    parser.add_argument(
        "--reference-mma-counts",
        help="file with one executed MMA count per matched reference dispatch, in trace order; "
        "without it the reference side is reported per wave instead of per MMA",
    )
    args = parser.parse_args(argv)

    try:
        pattern = re.compile(r"\bgemm_mmq_kernel\b")
        cells = parse_cells(args.cells)
        instructions_db = Database(args.instructions_db)
        stalls_db = Database(args.stalls_db)
        instructions = instructions_db.counters(set(INSTRUCTION_COUNTERS))
        stalls = stalls_db.counters(set(STALL_COUNTERS))
        dispatches = instructions_db.dispatches(pattern)
        paired = pair_cells(cells, dispatches)
        measured = [
            (cell, cell_metrics(cell, group, blocks, instructions, stalls))
            for cell, group, blocks in paired
        ]

        print("== rocprofv3 reduction")
        print(f"cells with an mmq route: {len(cells)}")
        print(f"gemm_mmq dispatches: {len(dispatches)}")
        print(
            f"submission denominator: {SWEEP_SUBMISSIONS_PER_CELL} per cell "
            "(two warmups plus five timed)"
        )
        print(f"rocpd schema: {instructions_db.schema_version()}")
        padded = sum(1 for cell, metrics in measured if metrics["padded"])
        if padded:
            print(
                f"note: {padded} of {len(cells)} cells do not fill the "
                f"{MMQ_TILE_ROWS}x{MMQ_TILE_OUT} tile; their per-MMA figures use "
                "the executed MMA count, not the useful-work count"
            )

        cell, group, blocks = focus_cell(cells, paired)
        metrics = cell_metrics(cell, group, blocks, instructions, stalls)
        print(
            f"\n== section 2.4 cell: {cell['dtype']} rows={cell['rows']} "
            f"n_in={cell['n_in']} n_out={cell['n_out']} "
            f"({metrics['dispatches']} dispatches, "
            f"{metrics['mmas_executed']} executed MMAs)"
        )
        reference = []
        if args.reference_instructions_db or args.reference_stalls_db:
            if not (args.reference_instructions_db and args.reference_stalls_db):
                raise Fail(
                    "--reference-instructions-db and --reference-stalls-db must be "
                    "given together"
                )
            reference = reference_rows(
                args.reference_instructions_db,
                args.reference_stalls_db,
                re.compile(args.reference_kernel_regex),
                args.reference_mma_counts,
            )
        reference_metrics = aggregate_reference_rows(reference)
        table = []
        for key, label in METRICS:
            current = metrics.get(key)
            recorded = RECORDED_YUUKA.get(key)
            if current is not None and recorded:
                delta = f"{100.0 * (current - recorded) / recorded:+.1f}%"
            else:
                delta = "na"
            table.append(
                [
                    label,
                    format_value(current),
                    format_value(recorded),
                    delta,
                    format_value(reference_metrics.get(key)),
                    format_value(RECORDED_LLAMA.get(key)),
                ]
            )
        print_table(
            "measured / recorded comparison",
            table,
            ("metric", "yuuka", "recorded", "delta", "reference", "recorded"),
        )
        if not reference:
            print(
                "\nreference column: not measured (no reference databases); recorded "
                "values are the 2026-11-18 llama.cpp attribution"
            )
        else:
            if len(reference) > 1:
                print(
                    f"\nreference column aggregates {len(reference)} kernel names; "
                    "pass a narrower --reference-kernel-regex to isolate one "
                    "template instantiation"
                )
            if reference_metrics.get("valu_per_mma") is None:
                print(
                    "\nreference per-MMA columns: no --reference-mma-counts given, so "
                    "the reference side is normalized per wave "
                    f"(VALU/wave {format_value(reference_metrics.get('valu_per_wave'))}, "
                    f"LDS/wave {format_value(reference_metrics.get('lds_per_wave'))})"
                )
        print(
            "\ndelta is against the 2026-11-18 attribution in docs/benchmarks/qwen3.5_4b.md; "
            "achieved TF/s also moves with clock state and code "
            "revision, so only the counters are expected to match."
        )

        print("\n== per-cell counters (CSV)")
        print("# " + ",".join(CSV_COLUMNS))
        for cell, metrics in measured:
            values = {
                "engine": "yuuka",
                "dtype": cell["dtype"],
                "rows": str(cell["rows"]),
                "n_out": str(cell["n_out"]),
                "n_in": str(cell["n_in"]),
                "kernel": "gemm_mmq_kernel",
                "dispatches": str(metrics["dispatches"]),
                "mmas_useful": str(metrics["mmas_useful"]),
                "mmas_executed": str(metrics["mmas_executed"]),
                "valu_per_mma": format_value(metrics["valu_per_mma"], 3),
                "lds_per_mma": format_value(metrics["lds_per_mma"], 3),
                "lds_bank_conflict_pct": format_value(metrics["lds_bank_conflict_pct"]),
                "alu_stalled_by_lds_pct": format_value(metrics["alu_stalled_by_lds_pct"]),
                "lds_latency_cycles": format_value(metrics["lds_latency_cycles"]),
                "occupancy_pct": format_value(metrics["occupancy_pct"]),
                "waves_per_workgroup": format_value(metrics["waves_per_workgroup"]),
                "achieved_tf_s": format_value(metrics["achieved_tf_s"]),
            }
            print(",".join(values[column] for column in CSV_COLUMNS))
        for row in reference:
            metrics = row["metrics"]
            values = {
                "engine": "llama.cpp",
                "dtype": "",
                "rows": "",
                "n_out": "",
                "n_in": "",
                "kernel": row["kernel"].replace(",", " "),
                "dispatches": str(metrics["dispatches"]),
                "mmas_useful": "",
                "mmas_executed": format_value(metrics["mmas_executed"]),
                "valu_per_mma": format_value(metrics["valu_per_mma"], 3),
                "lds_per_mma": format_value(metrics["lds_per_mma"], 3),
                "lds_bank_conflict_pct": format_value(metrics["lds_bank_conflict_pct"]),
                "alu_stalled_by_lds_pct": format_value(metrics["alu_stalled_by_lds_pct"]),
                "lds_latency_cycles": format_value(metrics["lds_latency_cycles"]),
                "occupancy_pct": format_value(metrics["occupancy_pct"]),
                "waves_per_workgroup": format_value(metrics["waves_per_workgroup"]),
                "achieved_tf_s": "",
            }
            print(",".join(values[column] for column in CSV_COLUMNS))
    except Fail as error:
        print(f"prefill_counters: error: {error}", file=sys.stderr)
        return 2
    return 0


def aggregate_reference_rows(rows):
    """Combine per-kernel reference rows into one comparison column."""
    if not rows:
        return {}
    dispatches = sum(row["metrics"]["dispatches"] for row in rows)
    weighted = defaultdict(float)
    for row in rows:
        metrics = row["metrics"]
        count = metrics["dispatches"]
        if metrics.get("mmas_executed"):
            weighted["mmas_executed"] += metrics["mmas_executed"]
            weighted["valu"] += metrics["valu_per_mma"] * metrics["mmas_executed"]
            weighted["lds"] += metrics["lds_per_mma"] * metrics["mmas_executed"]
        if metrics.get("valu_per_wave") is not None:
            weighted["valu_per_wave"] += metrics["valu_per_wave"] * count
            weighted["lds_per_wave"] += metrics["lds_per_wave"] * count
        for key in (
            "lds_bank_conflict_pct",
            "alu_stalled_by_lds_pct",
            "lds_latency_cycles",
            "occupancy_pct",
        ):
            weighted[key] += (metrics.get(key) or 0.0) * count
        weighted["waves"] += metrics.get("waves") or 0.0
        weighted["workgroups"] += metrics.get("workgroups") or 0.0
    combined = {}
    for key in (
        "lds_bank_conflict_pct",
        "alu_stalled_by_lds_pct",
        "lds_latency_cycles",
        "occupancy_pct",
    ):
        combined[key] = weighted[key] / dispatches if dispatches else None
    combined["waves_per_workgroup"] = (
        weighted["waves"] / weighted["workgroups"] if weighted["workgroups"] else None
    )
    combined["valu_per_wave"] = (
        weighted["valu_per_wave"] / dispatches if dispatches else None
    )
    combined["lds_per_wave"] = weighted["lds_per_wave"] / dispatches if dispatches else None
    combined["dispatches"] = dispatches
    if weighted["mmas_executed"]:
        combined["mmas_executed"] = int(weighted["mmas_executed"])
        combined["valu_per_mma"] = weighted["valu"] / weighted["mmas_executed"]
        combined["lds_per_mma"] = weighted["lds"] / weighted["mmas_executed"]
    else:
        combined["mmas_executed"] = None
        combined["valu_per_mma"] = None
        combined["lds_per_mma"] = None
    return combined


if __name__ == "__main__":
    sys.exit(main())
