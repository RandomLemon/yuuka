// Package npu reports XDNA NPU availability for opt-in heterogeneous
// experiments. No NPU execution kernel is shipped in the first release; every
// caller must fall back to the GPU or CPU path when availability or coverage is
// incomplete.
package npu

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// State classifies NPU readiness.
type State string

const (
	// Unavailable means no usable NPU device or driver was found.
	Unavailable State = "unavailable"
	// DeviceOnly means the kernel driver and device node exist but no userspace
	// runtime was found.
	DeviceOnly State = "device_only"
	// RuntimeOnly means a userspace runtime was found without an accessible
	// device node.
	RuntimeOnly State = "runtime_only"
	// Ready means device, driver, and userspace runtime were all found.
	Ready State = "ready"
)

// Capabilities is one point-in-time NPU probe result.
type Capabilities struct {
	State         State  `json:"state"`
	DevicePath    string `json:"device_path,omitempty"`
	DriverLoaded  bool   `json:"driver_loaded"`
	DevicePresent bool   `json:"device_present"`
	Userspace     bool   `json:"userspace"`
	RuntimePath   string `json:"runtime_path,omitempty"`
	Reason        string `json:"reason,omitempty"`
}

// Fallback reports whether callers must use the GPU or CPU path. It is true for
// every state in this release because NPU execution is not implemented.
func (c Capabilities) Fallback() bool { return true }

// Options overrides filesystem and lookup locations for tests.
type Options struct {
	DevRoot       string
	SysModuleRoot string
	OptRoot       string
	ExtraLibDirs  []string
	LookPath      func(string) (string, error)
}

func defaultOptions() Options {
	return Options{
		DevRoot:       "/dev/accel",
		SysModuleRoot: "/sys/module",
		OptRoot:       "/opt",
		ExtraLibDirs: []string{
			"/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu",
			"/opt/xilinx/xrt/lib",
		},
		LookPath: exec.LookPath,
	}
}

// Probe inspects the target host for an XDNA device, its driver, and the XRT
// userspace runtime. It never opens or initializes the device.
func Probe() Capabilities { return ProbeWith(defaultOptions()) }

// ProbeWith is Probe with explicit locations. It is intended for tests and
// packaging diagnostics.
func ProbeWith(opts Options) Capabilities {
	if opts.DevRoot == "" {
		opts.DevRoot = "/dev/accel"
	}
	if opts.SysModuleRoot == "" {
		opts.SysModuleRoot = "/sys/module"
	}
	if opts.LookPath == nil {
		opts.LookPath = exec.LookPath
	}
	caps := Capabilities{}
	if entries, err := os.ReadDir(opts.DevRoot); err == nil {
		for _, entry := range entries {
			if strings.HasPrefix(entry.Name(), "accel") {
				caps.DevicePresent = true
				caps.DevicePath = filepath.Join(opts.DevRoot, entry.Name())
				break
			}
		}
	}
	if _, err := os.Stat(filepath.Join(opts.SysModuleRoot, "amdxdna")); err == nil {
		caps.DriverLoaded = true
	}
	if path, err := opts.LookPath("xrt-smi"); err == nil {
		caps.Userspace = true
		caps.RuntimePath = path
	}
	if !caps.Userspace {
		for _, dir := range opts.ExtraLibDirs {
			if dir == "" {
				continue
			}
			matches, _ := filepath.Glob(filepath.Join(dir, "libxrt_coreutil*.so*"))
			if len(matches) > 0 {
				caps.Userspace = true
				caps.RuntimePath = matches[0]
				break
			}
			matches, _ = filepath.Glob(filepath.Join(dir, "libxrt_core*.so*"))
			if len(matches) > 0 {
				caps.Userspace = true
				caps.RuntimePath = matches[0]
				break
			}
		}
	}
	if opts.OptRoot != "" {
		if matches, _ := filepath.Glob(filepath.Join(opts.OptRoot, "xilinx", "xrt", "lib", "libxrt_coreutil*.so*")); len(matches) > 0 {
			caps.Userspace = true
			caps.RuntimePath = matches[0]
		}
	}
	switch {
	case caps.DevicePresent && caps.DriverLoaded && caps.Userspace:
		caps.State = Ready
	case caps.DevicePresent && caps.DriverLoaded:
		caps.State = DeviceOnly
		caps.Reason = "device and amdxdna driver are present, but no XRT userspace runtime was found"
	case caps.Userspace:
		caps.State = RuntimeOnly
		caps.Reason = "an XRT runtime was found, but no accessible /dev/accel device was found"
	default:
		caps.State = Unavailable
		caps.Reason = "no XDNA device, driver, or XRT userspace runtime was found"
	}
	if runtime.GOARCH != "amd64" {
		caps.Reason = strings.TrimSpace(caps.Reason + " (target architecture is not amd64)")
	}
	return caps
}

// Request resolves an opt-in NPU request. The returned selection is always
// "selected" in this release because no NPU execution path is implemented; the
// caller keeps its GPU or CPU backend. The string explains the fallback.
func Request(caps Capabilities) (device string, reason string) {
	if caps.State != Ready {
		return "selected", "NPU unavailable: " + caps.Reason + "; using the selected backend"
	}
	return "selected", "NPU runtime detected, but NPU execution is not implemented in this build; using the selected backend"
}
