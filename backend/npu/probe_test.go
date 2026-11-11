package npu

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func testOptions(t *testing.T) Options {
	t.Helper()
	root := t.TempDir()
	opts := Options{
		DevRoot:       filepath.Join(root, "dev"),
		SysModuleRoot: filepath.Join(root, "sys"),
		OptRoot:       filepath.Join(root, "opt"),
		LookPath:      func(string) (string, error) { return "", errors.New("not found") },
	}
	return opts
}

func TestProbeUnavailable(t *testing.T) {
	caps := ProbeWith(testOptions(t))
	if caps.State != Unavailable {
		t.Fatalf("state %q, want unavailable", caps.State)
	}
	if !caps.Fallback() {
		t.Fatal("unavailable NPU must fall back")
	}
	device, reason := Request(caps)
	if device != "selected" || reason == "" {
		t.Fatalf("request returned (%q,%q)", device, reason)
	}
}

func TestProbeDeviceOnly(t *testing.T) {
	opts := testOptions(t)
	if err := os.MkdirAll(opts.DevRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(opts.DevRoot, "accel0"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(opts.SysModuleRoot, "amdxdna"), 0o755); err != nil {
		t.Fatal(err)
	}
	caps := ProbeWith(opts)
	if caps.State != DeviceOnly {
		t.Fatalf("state %q, want device_only", caps.State)
	}
	if caps.DevicePath == "" || !caps.DriverLoaded || caps.Userspace {
		t.Fatalf("unexpected capabilities: %+v", caps)
	}
}

func TestProbeReady(t *testing.T) {
	opts := testOptions(t)
	if err := os.MkdirAll(opts.DevRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(opts.DevRoot, "accel0"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(opts.SysModuleRoot, "amdxdna"), 0o755); err != nil {
		t.Fatal(err)
	}
	opts.LookPath = func(name string) (string, error) {
		if name == "xrt-smi" {
			return "/usr/bin/xrt-smi", nil
		}
		return "", errors.New("not found")
	}
	caps := ProbeWith(opts)
	if caps.State != Ready {
		t.Fatalf("state %q, want ready", caps.State)
	}
	if device, _ := Request(caps); device != "selected" {
		t.Fatalf("ready NPU must still fall back in this release, got %q", device)
	}
}
