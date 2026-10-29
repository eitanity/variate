package suite

import (
	"bytes"
	"context"
	"errors"
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	"code.eitanity.com/variate"
)

// collector keeps every delivered round.
type collector struct {
	rounds []variate.Round
	stopAt int // return ErrStop at this Seq, −1 for never
	sleep  time.Duration
	mu     sync.Mutex
}

func (c *collector) Observe(r *variate.Round) error {
	if c.sleep > 0 {
		time.Sleep(c.sleep)
	}
	cp := *r
	cp.Steps = append([]variate.Step(nil), r.Steps...)
	c.mu.Lock()
	c.rounds = append(c.rounds, cp)
	c.mu.Unlock()
	if c.stopAt >= 0 && r.Seq == uint64(c.stopAt) {
		return variate.ErrStop
	}
	return nil
}

func runStream(t *testing.T, g variate.Graph, cfg variate.Config, seed uint64) (*variate.Compiled, *variate.Workspace, *variate.Result) {
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
	r, err := w.Run(context.Background(), seed)
	if err != nil {
		t.Fatal(err)
	}
	return m, w, r
}

// TestStreamCompletenessAndPaths: with an empty Watch and no delays, the
// delivered rounds per source equal Events, each period's values sum to
// the period total, leaf counts equal the terminal counts, and every path
// is consistent.
func TestStreamCompletenessAndPaths(t *testing.T) {
	g := genGraph(t, 4, false)
	col := &collector{stopAt: -1}
	const h = 400
	m, _, r := runStream(t, g, variate.Config{Periods: h, BlockPeriods: 50, Series: true, Observer: col, Workers: 3}, 8)
	sh := shapeOf(m)
	perSource := make([]uint64, len(r.Sources))
	perLeaf := make([]uint64, len(r.Nodes))
	perPeriod := make([]float64, h)
	for i, rd := range col.rounds {
		if rd.Seq != uint64(i) {
			t.Fatalf("seq %d at position %d", rd.Seq, i)
		}
		perSource[rd.Source]++
		perLeaf[rd.Leaf]++
		perPeriod[rd.Period] += rd.Value
		if i > 0 {
			prev := col.rounds[i-1]
			if rd.Period < prev.Period || (rd.Period == prev.Period && rd.Arrival < prev.Arrival) {
				t.Fatalf("order broken at %d: %v after %v", i, rd.Period, prev.Period)
			}
		}
		sum := 0.0
		for s, st := range rd.Steps {
			sum += st.Value
			if s > 0 && st.T < rd.Steps[s-1].T {
				t.Fatalf("round %d: step times decrease", i)
			}
			if s < len(rd.Steps)-1 {
				if st.Outcome < 0 || sh.outTarget[int(st.Outcome)] != int(rd.Steps[s+1].Node) {
					t.Fatalf("round %d: outcome %d does not lead to the next step", i, st.Outcome)
				}
			} else if st.Outcome != -1 || st.Node != rd.Leaf {
				t.Fatalf("round %d: last step %+v leaf %d", i, st, rd.Leaf)
			}
		}
		if math.Abs(sum-rd.Value) > 1e-9*math.Max(1, rd.Value) {
			t.Fatalf("round %d: steps sum %g value %g", i, sum, rd.Value)
		}
		if rd.Steps[0].T != rd.Arrival || rd.Steps[len(rd.Steps)-1].T != rd.End {
			t.Fatalf("round %d: arrival %g first step %g, end %g last step %g", i, rd.Arrival, rd.Steps[0].T, rd.End, rd.Steps[len(rd.Steps)-1].T)
		}
	}
	for k := range r.Sources {
		if perSource[k] != r.Events[k] {
			t.Errorf("source %s: delivered %d events %d", r.Sources[k], perSource[k], r.Events[k])
		}
	}
	for v := range r.Nodes {
		if sh.leaf[v] {
			var reach uint64
			for k := range r.Sources {
				reach += r.Reach[k][v]
			}
			if perLeaf[v] != reach {
				t.Errorf("leaf %s: delivered %d reach %d", r.Nodes[v], perLeaf[v], reach)
			}
		}
	}
	for p := int64(0); p < h; p++ {
		if math.Abs(perPeriod[p]-r.Series.Total(p)) > 1e-9*math.Max(1, r.Series.Total(p)) {
			t.Errorf("period %d: rounds sum %g total %g", p, perPeriod[p], r.Series.Total(p))
		}
	}
}

// TestStreamFiltering: a Watch on one node delivers exactly the rounds
// that reached it; a leaf watch exactly those ending there; MinValue only
// the rounds worth at least that.
func TestStreamFiltering(t *testing.T) {
	g := Diamond()
	col := &collector{stopAt: -1}
	cfg := variate.Config{Periods: 2000, BlockPeriods: 100, Observer: col, Watch: variate.Watch{Nodes: []string{"b"}}}
	_, _, r := runStream(t, g, cfg, 2)
	if uint64(len(col.rounds)) != r.Reach[0][1] {
		t.Errorf("node watch: delivered %d reach %d", len(col.rounds), r.Reach[0][1])
	}
	col = &collector{stopAt: -1}
	cfg = variate.Config{Periods: 2000, BlockPeriods: 100, Observer: col, Watch: variate.Watch{Leaves: []string{"d"}, MinValue: 11}}
	_, _, r = runStream(t, g, cfg, 2)
	// Every round ends at d; those through b book more than 10.
	if uint64(len(col.rounds)) != r.Reach[0][1] {
		t.Errorf("min-value watch: delivered %d, rounds through b %d", len(col.rounds), r.Reach[0][1])
	}
	for _, rd := range col.rounds {
		if rd.Value < 11 || rd.Leaf != 3 {
			t.Fatalf("delivered %+v", rd)
		}
	}
}

// TestStreamNoEffectAndDeterminism: the Result is bit-identical with no
// observer, a Counter, a collector and a slow observer; the delivered
// bytes are identical across worker counts.
func TestStreamNoEffectAndDeterminism(t *testing.T) {
	g := genGraph(t, 6, true)
	cfg := variate.Config{Periods: 600, BlockPeriods: 50, Workers: 4}
	_, _, base := runStream(t, g, cfg, 5)
	ref := fingerprint(base)
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	observers := []variate.Observer{variate.NewCounter(m), &collector{stopAt: -1}, &collector{stopAt: -1, sleep: 50 * time.Microsecond}}
	for i, o := range observers {
		c := cfg
		c.Observer = o
		_, _, r := runStream(t, g, c, 5)
		for f, x := range fingerprint(r) {
			if x != ref[f] {
				t.Fatalf("observer %d changes the result at field %d", i, f)
			}
		}
	}
	var refBytes []byte
	for _, workers := range []int{1, 2, 7, 32} {
		var buf bytes.Buffer
		c := cfg
		c.Workers = workers
		c.Observer = variate.NewRecordWriter(&buf, m.LongestPath())
		runStream(t, g, c, 5)
		if refBytes == nil {
			refBytes = buf.Bytes()
			if len(refBytes) == 0 {
				t.Fatal("nothing delivered")
			}
			continue
		}
		if !bytes.Equal(buf.Bytes(), refBytes) {
			t.Errorf("%d workers: delivered bytes differ", workers)
		}
	}
}

// TestStreamStop: ErrStop on round m delivers exactly rounds 0..m, and the
// Result covers whole blocks up to that one.
func TestStreamStop(t *testing.T) {
	col := &collector{stopAt: 137}
	_, _, r := runStream(t, Diamond(), variate.Config{Periods: 5000, BlockPeriods: 100, Observer: col, Workers: 3}, 4)
	if len(col.rounds) != 138 || col.rounds[137].Seq != 137 {
		t.Errorf("delivered %d rounds", len(col.rounds))
	}
	if !r.Stopped || r.Converged || r.Periods%100 != 0 || r.Periods == 0 || r.Periods == 5000 {
		t.Errorf("stopped %v converged %v periods %d", r.Stopped, r.Converged, r.Periods)
	}
	// The result covers exactly the finalised blocks: events of the last
	// delivered round's block are included.
	if int64(col.rounds[137].Period) >= r.Periods {
		t.Errorf("round 137 in period %d beyond the result's %d periods", col.rounds[137].Period, r.Periods)
	}
}

type observerFunc func(r *variate.Round) error

func (f observerFunc) Observe(r *variate.Round) error { return f(r) }

func TestStreamScratchOverflow(t *testing.T) {
	cfg := variate.Config{Periods: 10, RoundScratch: 2, Observer: &collector{stopAt: -1}}
	m, _, err := Diamond().Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	_, err = w.Run(context.Background(), 1)
	if err == nil || !strings.Contains(err.Error(), "round scratch full") {
		t.Errorf("got %v", err)
	}
}

// TestStreamRecordFile: a tiny record buffer forces records into the
// slot files; the delivered sequence is unchanged.
func TestStreamRecordFile(t *testing.T) {
	g := Diamond()
	a := &collector{stopAt: -1}
	runStream(t, g, variate.Config{Periods: 300, BlockPeriods: 50, Observer: a}, 9)
	b := &collector{stopAt: -1}
	runStream(t, g, variate.Config{Periods: 300, BlockPeriods: 50, Observer: b, RecordBuffer: 1}, 9)
	if len(a.rounds) != len(b.rounds) || len(a.rounds) == 0 {
		t.Fatalf("delivered %d vs %d", len(a.rounds), len(b.rounds))
	}
	for i := range a.rounds {
		if a.rounds[i].Seq != b.rounds[i].Seq || a.rounds[i].Arrival != b.rounds[i].Arrival || a.rounds[i].Value != b.rounds[i].Value {
			t.Fatalf("round %d differs", i)
		}
	}
}

// TestStreamObserverErrors: an error fails the run, a panic fails the run
// without crashing, and the workspace stays usable.
func TestStreamObserverErrors(t *testing.T) {
	boom := errors.New("boom")
	cfg := variate.Config{Periods: 100, BlockPeriods: 10, Observer: observerFunc(func(*variate.Round) error { return boom })}
	m, _, err := Diamond().Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), 1); !errors.Is(err, boom) {
		t.Errorf("got %v", err)
	}
	cfg.Observer = observerFunc(func(r *variate.Round) error { var none variate.Observer; return none.Observe(r) })
	m, _, _ = Diamond().Compile(cfg)
	w2, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()
	if _, err := w2.Run(context.Background(), 1); err == nil || !strings.Contains(err.Error(), "panicked") {
		t.Errorf("got %v", err)
	}
	cfg.Observer = variate.NewCounter(m)
	w3, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w3.Close()
	if _, err := w3.Run(context.Background(), 1); err != nil {
		t.Errorf("usable after: %v", err)
	}
}
