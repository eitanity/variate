package suite

import (
	"context"
	"math"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

func newWorkspace(t *testing.T, g variate.Graph, cfg variate.Config) *variate.Workspace {
	t.Helper()
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(w.Close)
	return w
}

func fingerprint(r *variate.Result) []float64 {
	fp := make([]float64, 0, 16+2*len(r.ByCategory)+2*len(r.EntriesPer))
	fp = append(fp, r.Total.Sum.Value(), r.Total.W.Mean, r.Total.W.M2, r.Total.Block.Mean, r.Total.Min, r.Total.Max, float64(r.Events[0]), float64(r.Steps), r.BeyondHorizon)
	for _, a := range r.ByCategory {
		fp = append(fp, a.Sum.Value(), a.W.M2)
	}
	for v := range r.EntriesPer {
		fp = append(fp, r.EntriesPer[v].Sum.Value(), float64(r.Reach[0][v]))
	}
	q, _ := r.Total.Hist.Quantile(0.99)
	fp = append(fp, q, float64(r.Total.Hist.AtOrAbove(r.Total.Mean())))
	return fp
}

// TestWorkspaceDeterminismAcrossWorkers: the same seed gives bit-identical
// results for 1, 2, 7 and 32 workers, with and without delays.
func TestWorkspaceDeterminismAcrossWorkers(t *testing.T) {
	for _, delays := range []bool{false, true} {
		g := genGraph(t, 5, delays)
		var ref []float64
		for _, workers := range []int{1, 2, 7, 32} {
			cfg := variate.Config{Periods: 6000, BlockPeriods: 100, Workers: workers, Batch: 8}
			w := newWorkspace(t, g, cfg)
			r, err := w.Run(context.Background(), 42)
			if err != nil {
				t.Fatal(err)
			}
			fp := fingerprint(r)
			if ref == nil {
				ref = fp
				continue
			}
			for i := range fp {
				if fp[i] != ref[i] {
					t.Fatalf("delays %v: %d workers differ from 1 at field %d: %v vs %v", delays, workers, i, fp[i], ref[i])
				}
			}
		}
	}
}

// TestWorkspaceNoAllocation: Run allocates nothing and triggers no GC on a
// warm workspace, with and without delays.
func TestWorkspaceNoAllocation(t *testing.T) {
	for _, delays := range []bool{false, true} {
		g := genGraph(t, 7, delays)
		w := newWorkspace(t, g, variate.Config{Periods: 3000, BlockPeriods: 200, Workers: 4})
		ctx := context.Background()
		if _, err := w.Run(ctx, 1); err != nil {
			t.Fatal(err)
		}
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		for i := 0; i < 10; i++ {
			if _, err := w.Run(ctx, uint64(i)); err != nil {
				t.Fatal(err)
			}
		}
		runtime.ReadMemStats(&after)
		if after.NumGC != before.NumGC {
			t.Errorf("delays %v: GC ran %d times during 10 runs", delays, after.NumGC-before.NumGC)
		}
		if after.Mallocs-before.Mallocs > 10 {
			t.Errorf("delays %v: %d allocations during 10 runs", delays, after.Mallocs-before.Mallocs)
		}
	}
}

func TestWorkspaceFootprintRefused(t *testing.T) {
	g := Diamond()
	m, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = m.NewWorkspace(variate.Config{Periods: 1000, MaxMemory: 1 << 20, Workers: 8})
	if err == nil || !strings.Contains(err.Error(), "exceeds MaxMemory") {
		t.Errorf("got %v", err)
	}
}

func TestWorkspaceCancel(t *testing.T) {
	w := newWorkspace(t, Diamond(), variate.Config{Periods: 20000, BlockPeriods: 1000, Workers: 2})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := w.Run(ctx, 1); err == nil || !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("got %v", err)
	}
	// The workspace is still usable.
	if r, err := w.Run(context.Background(), 1); err != nil || r.Periods != 20000 {
		t.Errorf("run after a cancelled one: %v", err)
	}
}

func TestWorkspaceClosed(t *testing.T) {
	w := newWorkspace(t, Diamond(), variate.Config{Periods: 100})
	w.Close()
	w.Close()
	if _, err := w.Run(context.Background(), 1); err == nil {
		t.Error("run on a closed workspace")
	}
}

// holdingObserver holds the finaliser in its first delivery until
// released, so that the run is certainly in progress.
type holdingObserver struct {
	entered, release chan struct{}
	once             sync.Once
}

func newHoldingObserver() *holdingObserver {
	return &holdingObserver{entered: make(chan struct{}), release: make(chan struct{})}
}

func (h *holdingObserver) Observe(*variate.Round) error {
	h.once.Do(func() {
		close(h.entered)
		<-h.release
	})
	return nil
}

// TestWorkspaceOneRunAtATime: a Run while another is in progress is
// refused, and the first completes as if it had not been.
func TestWorkspaceOneRunAtATime(t *testing.T) {
	obs := newHoldingObserver()
	w := newWorkspace(t, Diamond(), variate.Config{Periods: 2000, BlockPeriods: 10, Workers: 4, Observer: obs})
	done := make(chan error)
	go func() {
		_, err := w.Run(context.Background(), 1)
		done <- err
	}()
	<-obs.entered
	if _, err := w.Run(context.Background(), 2); err == nil || !strings.Contains(err.Error(), "already in progress") {
		t.Errorf("second run: %v", err)
	}
	close(obs.release)
	if err := <-done; err != nil {
		t.Errorf("first run: %v", err)
	}
	if _, err := w.Run(context.Background(), 3); err != nil {
		t.Errorf("run after: %v", err)
	}
}

// TestWorkspaceCloseDuringRun: Close during a Run ends it with an error
// once the pool has stopped, instead of leaving it waiting, and leaks
// nothing.
func TestWorkspaceCloseDuringRun(t *testing.T) {
	g := guard(t)
	obs := newHoldingObserver()
	cfg := variate.Config{Periods: 2000, BlockPeriods: 10, Workers: 4, Observer: obs}
	m, _, err := Diamond().Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error)
	go func() {
		_, err := w.Run(context.Background(), 1)
		done <- err
	}()
	<-obs.entered
	closed := make(chan struct{})
	go func() {
		w.Close()
		close(closed)
	}()
	// Once Close has begun, a further Run says the workspace is closed.
	// bound: until Close has begun
	for {
		_, err := w.Run(context.Background(), 2)
		if err != nil && strings.Contains(err.Error(), "closed") {
			break
		}
		time.Sleep(time.Millisecond) //nolint:forbidigo // waiting for Close to begin
	}
	close(obs.release)
	if err := <-done; err == nil || !strings.Contains(err.Error(), "closed during the run") {
		t.Errorf("run: %v", err)
	}
	<-closed
	g.check(t, "close during a run")
}

func TestSummarize(t *testing.T) {
	g := Diamond()
	w := newWorkspace(t, g, variate.Config{Periods: 20000, NodeHist: true})
	r, err := w.Run(context.Background(), 3)
	if err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	w.Summarize(&s) // reuse
	if s.Total.N != 20000 || s.Total.Mean != r.Total.Mean() || !(s.Total.P50 > 0) || !(s.Total.P999 >= s.Total.P99) || s.Total.Clipped {
		t.Errorf("total %+v", s.Total)
	}
	share := s.ByCategory["labour"].Share + s.ByCategory["parts"].Share
	if math.Abs(share-1) > 1e-9 {
		t.Errorf("category shares sum to %g", share)
	}
	if math.Abs(s.BySource["s"].Share-1) > 1e-9 {
		t.Errorf("source share %g", s.BySource["s"].Share)
	}
	if p := s.ReachProb["s"]["d"]; p.P != 1 || p.StdErr != 0 {
		t.Errorf("reach d %+v", p)
	}
	if p := s.ReachProb["s"]["b"]; math.Abs(p.P-0.3) > 5*p.StdErr {
		t.Errorf("reach b %+v", p)
	}
	if p := s.Terminal["s"]["d"]; p.P != 1 {
		t.Errorf("terminal %+v", p)
	}
	if _, ok := s.Terminal["s"]["a"]; ok {
		t.Error("non-leaf in terminal")
	}
	if p := s.OutcomeP["a→c"]; math.Abs(p.P-0.7) > 5*p.StdErr {
		t.Errorf("outcome a→c %+v", p)
	}
	// The curve starts at the histogram's lower edge, so its first point is
	// the share of periods with a non-zero total, and it never rises.
	first := 1 - float64(r.Total.Zero)/float64(r.Total.N())
	if len(s.Exceedance) < 2 || math.Abs(s.Exceedance[0].P-first) > 1e-12 {
		t.Errorf("exceedance %v … %v, non-zero share %g", s.Exceedance[0], s.Exceedance[len(s.Exceedance)-1], first)
	}
	for i := 1; i < len(s.Exceedance); i++ {
		if s.Exceedance[i].P > s.Exceedance[i-1].P || s.Exceedance[i].X <= s.Exceedance[i-1].X {
			t.Fatalf("exceedance not monotone at %d: %v %v", i, s.Exceedance[i-1], s.Exceedance[i])
		}
	}
	if e := s.EntriesPer["d"]; math.Abs(e.Mean-r.EntriesPer[3].Mean()) > 0 {
		t.Errorf("entries %+v", e)
	}
	bands := w.Bands([]float64{0, 10, 1e9}, make([]float64, 2))
	if math.Abs(bands[0]+bands[1]-1) > 1e-12 {
		t.Errorf("bands %v", bands)
	}
}

// TestFixedRunBlocks: a fixed run without a block setting gets enough
// blocks for the batch-means error of a model with delays, an explicit
// block setting is kept, and an adaptive run keeps the default.
func TestFixedRunBlocks(t *testing.T) {
	m, err := config.Load("../../examples/delays.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg := variate.Config{Periods: 20000, Workers: 2}
	c, _, err := m.Graph.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 4)
	if err != nil {
		t.Fatal(err)
	}
	if se := r.Total.StdErr(true); !(se > 0) {
		t.Errorf("standard error %g after %d periods", se, r.Periods)
	}
	var sum variate.Summary
	w.Summarize(&sum)
	for _, row := range sum.Plan {
		if row.Criterion == "target-rse" && row.Driver == "total" && (row.PeriodsNeeded != 0 || math.IsInf(row.Observed, 1)) {
			t.Errorf("target-rse total: planned %d observed %g", row.PeriodsNeeded, row.Observed)
		}
	}
	cfg.BlockPeriods = 4096
	w2, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()
	r2, err := w2.Run(context.Background(), 4)
	if err != nil {
		t.Fatal(err)
	}
	if se := r2.Total.StdErr(true); !math.IsNaN(se) {
		t.Errorf("explicit 4096-period blocks: standard error %g from %d periods", se, r2.Periods)
	}
}
