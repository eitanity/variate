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
