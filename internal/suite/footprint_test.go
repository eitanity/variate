package suite

import (
	"runtime"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// heapInUse forces a collection and reads the live heap.
func heapInUse() uint64 {
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return ms.HeapInuse
}

// TestFootprint: the planned footprint matches the heap a workspace
// takes, within a quarter, for the pump-station model and a 1000-node
// graph, and MaxMemory below the plan is refused with the breakdown.
func TestFootprint(t *testing.T) {
	m, err := config.Load("../../examples/pump_station.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		g    variate.Graph
		cfg  variate.Config
	}{
		{"pump station", m.Graph, variate.Config{Periods: 1000, Workers: 4}},
		{"1000 nodes", bigGraph(1000), variate.Config{Periods: 1000, Workers: 4}},
	} {
		c, _, err := tc.g.Compile(tc.cfg)
		if err != nil {
			t.Fatal(err)
		}
		before := heapInUse()
		w, err := c.NewWorkspace(tc.cfg)
		if err != nil {
			t.Fatal(err)
		}
		measured := float64(heapInUse() - before)
		planned := float64(w.Footprint())
		t.Logf("%s: planned %.1f MB, heap grew %.1f MB", tc.name, planned/1e6, measured/1e6)
		if measured > 1.25*planned || measured < 0.75*planned {
			t.Errorf("%s: planned %.1f MB but the heap grew %.1f MB", tc.name, planned/1e6, measured/1e6)
		}
		w.Close()
		small := tc.cfg
		small.MaxMemory = w.Footprint() - 1
		if _, err := c.NewWorkspace(small); err == nil || !strings.Contains(err.Error(), "exceeds MaxMemory") || !strings.Contains(err.Error(), "histograms") {
			t.Errorf("%s: refusal %v", tc.name, err)
		}
	}
}

// TestFootprintFitsWorkers: with Workers unset the workspace takes the
// most workers up to GOMAXPROCS that fit MaxMemory, so a limit that
// holds two of four gives two; a Workers set above the limit, or a
// limit below one worker's, is refused.
func TestFootprintFitsWorkers(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	c, _, err := bigGraph(1000).Compile(variate.Config{NoBreakdown: true})
	if err != nil {
		t.Fatal(err)
	}
	footprint := func(cfg variate.Config) (int64, int) {
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		return w.Footprint(), w.Workers()
	}
	one, _ := footprint(variate.Config{NoBreakdown: true, Periods: 1000, Workers: 1})
	two, _ := footprint(variate.Config{NoBreakdown: true, Periods: 1000, Workers: 2})
	if got, n := footprint(variate.Config{NoBreakdown: true, Periods: 1000}); n != 4 || got <= two {
		t.Fatalf("no limit: %d workers, %d bytes", n, got)
	}
	if got, n := footprint(variate.Config{NoBreakdown: true, Periods: 1000, MaxMemory: two}); n != 2 || got != two {
		t.Errorf("limit %d: %d workers, %d bytes; want 2 workers", two, n, got)
	}
	if _, err := c.NewWorkspace(variate.Config{NoBreakdown: true, Periods: 1000, Workers: 3, MaxMemory: two}); err == nil || !strings.Contains(err.Error(), "exceeds MaxMemory") {
		t.Errorf("3 workers set over the limit: %v", err)
	}
	if _, err := c.NewWorkspace(variate.Config{NoBreakdown: true, Periods: 1000, MaxMemory: one - 1}); err == nil || !strings.Contains(err.Error(), "exceeds MaxMemory") {
		t.Errorf("a limit below one worker: %v", err)
	}
}
