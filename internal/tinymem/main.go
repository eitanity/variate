// Command tinymem compiles a model and runs its transform with the
// browser's settings, built with TinyGo and run under a WASI runtime the
// way the page runs the engine, and fails when the heap reserved
// exceeds browser.MaxHeap: past it a doubling of the wasm memory can
// fail, and TinyGo then stops the engine instead of refusing.
package main

import (
	"fmt"
	"os"
	"runtime"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
	"code.eitanity.com/variate/internal/browser"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	if len(args) != 1 {
		fmt.Fprintln(os.Stderr, "usage: tinymem model.yaml")
		return 2
	}
	if err := check(args[0]); err != nil {
		fmt.Fprintln(os.Stderr, args[0]+": "+err.Error())
		return 1
	}
	return 0
}

func check(path string) error {
	m, err := config.Load(path)
	if err != nil {
		return err
	}
	browser.Apply(&m.Config)
	c, _, err := m.Graph.Compile(m.Config)
	if err != nil {
		return fmt.Errorf("compile: %w", err)
	}
	cfg := m.Config
	tr, err := c.Transform(variate.TransformConfig{TailP: cfg.TailP, MaxGrid: cfg.TransformGrid, Tol: cfg.TransformTol, MaxWork: cfg.TransformWork, MaxBytes: cfg.TransformBytes})
	if err != nil {
		return fmt.Errorf("transform: %w", err)
	}
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	outcome := fmt.Sprintf("%d points", tr.M)
	if !tr.Applicable {
		outcome = "refused: " + tr.Reason
	}
	fmt.Printf("%s: heap %d MB, transform %s\n", path, ms.HeapSys>>20, outcome)
	if ms.HeapSys > browser.MaxHeap {
		return fmt.Errorf("the heap reserved %d MB, above %d MB", ms.HeapSys>>20, browser.MaxHeap>>20)
	}
	return nil
}
