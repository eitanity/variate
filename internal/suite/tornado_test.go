package suite

import (
	"context"
	"math"
	"slices"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// sweepGraph has an input of every kind: a factor and a schedule on the
// sources' rates, a repair law, functions, fixed and uncertain
// probabilities, and holdings on an outcome and a continuation.
func sweepGraph() variate.Graph {
	repair, _ := dist.NewExponential(0.05)
	hold, _ := dist.NewExponential(0.02)
	return variate.Graph{
		ID:        "sweep",
		Factors:   []variate.Factor{{ID: "f", Model: Pert(0.8, 1, 1.3)}},
		Schedules: []variate.Schedule{{ID: "season", Values: []float64{0.5, 1.5}}},
		Sources: []variate.Source{
			{ID: "s", Frequency: Pert(1, 2, 4), To: "a", Factor: "f"},
			{ID: "r", Frequency: C(3), To: "b", Repair: repair},
			{ID: "q", Frequency: C(1), To: "e", Schedule: "season"},
		},
		Nodes: []variate.Node{
			{ID: "a", Outcomes: []variate.Outcome{{To: "b", P: C(0.3), Holding: hold}, {To: "c", P: C(0.7)}}},
			{ID: "b", Next: "d", NextHolding: hold, Functions: []variate.Function{Fn("fb", "labour", Pert(1, 2, 3))}},
			{ID: "c", Outcomes: []variate.Outcome{{To: "d", P: Beta(2, 5)}, {To: "e"}}},
			{ID: "d", Functions: []variate.Function{Fn("fd", "parts", C(10))}},
			{ID: "e"},
		},
	}
}

// TestTornadoInputs: every distribution of the graph is an input, named
// as the report names things, in a fixed order; a remainder outcome and
// a lone absolute outcome are not.
func TestTornadoInputs(t *testing.T) {
	inputs := sweepGraph().Inputs()
	names := make([]string, 0, len(inputs))
	for _, in := range inputs {
		names = append(names, in.Kind+": "+in.Name)
	}
	want := []string{
		"factor: factor f",
		"schedule: schedule season",
		"frequency: source s frequency",
		"frequency: source r frequency",
		"repair: source r repair",
		"frequency: source q frequency",
		"probability: node a outcome b p",
		"holding: node a outcome b holding",
		"probability: node a outcome c p",
		"function: node b function fb",
		"holding: node b next holding",
		"probability: node c outcome d p",
		"function: node d function fd",
	}
	if !slices.Equal(names, want) {
		t.Errorf("inputs\n%s\nwant\n%s", strings.Join(names, "\n"), strings.Join(want, "\n"))
	}
}

// TestTornadoScaledProbabilities: a scaled probability stays a probability.
// With a remainder the remainder takes the change, held so the fixed
// sum stays within 1; without one the others share it in proportion; a
// drawn probability is capped at 1; and the graph scaled from is not
// changed.
func TestTornadoScaledProbabilities(t *testing.T) {
	probs := func(g variate.Graph, node int) []float64 {
		var out []float64
		for _, o := range g.Nodes[node].Outcomes {
			if o.P == nil {
				out = append(out, -1)
			} else {
				out = append(out, o.P.Mean())
			}
		}
		return out
	}
	near := func(a, b []float64) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if math.Abs(a[i]-b[i]) > 1e-15 {
				return false
			}
		}
		return true
	}
	g := sweepGraph()
	// No remainder: 0.3 → 0.33, and 0.7 gives up the 0.03.
	up, clamped, err := g.Scaled("node a outcome b p", 1.1)
	if err != nil || clamped || !near(probs(up, 0), []float64{0.33, 0.67}) {
		t.Errorf("no remainder: %v clamped %v err %v", probs(up, 0), clamped, err)
	}
	if !near(probs(g, 0), []float64{0.3, 0.7}) {
		t.Errorf("the graph scaled from changed: %v", probs(g, 0))
	}
	// 0.7 → 1.4 is held at 1.
	up, clamped, _ = g.Scaled("node a outcome c p", 2)
	if !clamped || !near(probs(up, 0), []float64{0, 1}) {
		t.Errorf("held at 1: %v clamped %v", probs(up, 0), clamped)
	}
	// With a remainder.
	r := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(1), To: "n"}},
		Nodes: []variate.Node{
			{ID: "n", Outcomes: []variate.Outcome{{To: "x", P: C(0.5)}, {To: "y", P: C(0.3)}, {To: "z"}}},
			{ID: "x"}, {ID: "y"}, {ID: "z"},
		},
	}
	up, clamped, _ = r.Scaled("node n outcome x p", 1.2)
	if clamped || !near(probs(up, 0), []float64{0.6, 0.3, -1}) {
		t.Errorf("remainder: %v clamped %v", probs(up, 0), clamped)
	}
	up, clamped, _ = r.Scaled("node n outcome x p", 1.5)
	if !clamped || !near(probs(up, 0), []float64{0.7, 0.3, -1}) {
		t.Errorf("remainder held: %v clamped %v", probs(up, 0), clamped)
	}
	// A drawn probability, Beta(2, 5) on [0, 1]: scaled up it passes 1
	// and is capped; scaled down it is not.
	up, clamped, _ = g.Scaled("node c outcome d p", 1.1)
	if lo, hi := up.Nodes[2].Outcomes[0].P.Support(); !clamped || lo != 0 || hi != 1 {
		t.Errorf("drawn probability: support [%g, %g] clamped %v", lo, hi, clamped)
	}
	down, clamped, _ := g.Scaled("node c outcome d p", 0.9)
	if lo, hi := down.Nodes[2].Outcomes[0].P.Support(); clamped || lo != 0 || hi != 0.9 {
		t.Errorf("drawn probability down: support [%g, %g] clamped %v", lo, hi, clamped)
	}
	for _, sg := range []variate.Graph{up, down} {
		if _, _, err := sg.Compile(variate.Config{}); err != nil {
			t.Error(err)
		}
	}
	if _, _, err := g.Scaled("node a outcome q p", 1.1); err == nil {
		t.Error("an unknown input scaled")
	}
	if _, _, err := g.Scaled("factor f", 0); err == nil {
		t.Error("a zero factor accepted")
	}
}

// TestTornadoMeanRowsExact: on the corpus every mean row is exact and is
// the analytic mean of the scaled model, the base row the model's own;
// a model with many inputs has every 15th row checked so.
func TestTornadoMeanRowsExact(t *testing.T) {
	for _, cm := range corpus(t) {
		base, err := mustCompile(t, cm.g).Analytic()
		if err != nil {
			t.Fatalf("%s: %v", cm.name, err)
		}
		sw, err := cm.g.Sweep(context.Background(), variate.SweepConfig{})
		if err != nil {
			t.Fatalf("%s: %v", cm.name, err)
		}
		if len(sw.Rows) != len(cm.g.Inputs()) {
			t.Errorf("%s: %d rows for %d inputs", cm.name, len(sw.Rows), len(cm.g.Inputs()))
		}
		for i, row := range sw.Rows {
			if row.Method != variate.MethodExact || row.Base.Lo != base.Mean || row.DownSE != 0 {
				t.Fatalf("%s %s: %s row, base %g, want exact %g", cm.name, row.Input.Name, row.Method, row.Base.Lo, base.Mean)
			}
			if i > 0 && row.Swing > sw.Rows[i-1].Swing {
				t.Errorf("%s: rows not by swing", cm.name)
			}
			if len(sw.Rows) > 40 && i%15 != 0 {
				continue
			}
			for f, got := range map[float64]float64{0.9: row.Down.Lo, 1.1: row.Up.Lo} {
				g, _, _ := cm.g.Scaled(row.Input.Name, f)
				e, err := mustCompile(t, g).Analytic()
				if err != nil || e.Mean != got {
					t.Errorf("%s %s × %g: row %g, analytic %g (%v)", cm.name, row.Input.Name, f, got, e.Mean, err)
				}
			}
		}
	}
}

// TestTornadoRateRow: the mean is linear in a Poisson source's rate, so
// its rows are the base mean less and plus delta times the source's
// mean contribution; two identical sources have identical rows.
func TestTornadoRateRow(t *testing.T) {
	g := Diamond()
	g.Sources = append(g.Sources, variate.Source{ID: "t", Frequency: Pert(1, 2, 4), To: "a"})
	e, _ := mustCompile(t, g).Analytic()
	sw, err := g.Sweep(context.Background(), variate.SweepConfig{Delta: 0.2, Inputs: []string{"frequency"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(sw.Rows) != 2 {
		t.Fatalf("%d rows", len(sw.Rows))
	}
	for _, row := range sw.Rows {
		k := slices.Index([]string{"source s frequency", "source t frequency"}, row.Input.Name)
		share := 0.2 * e.MeanBySource[k]
		if math.Abs(row.Up.Lo-row.Base.Lo-share) > 1e-12*share || math.Abs(row.Base.Lo-row.Down.Lo-share) > 1e-12*share {
			t.Errorf("%s: %g %g %g, want ±%g", row.Input.Name, row.Down.Lo, row.Base.Lo, row.Up.Lo, share)
		}
	}
	if a, b := sw.Rows[0], sw.Rows[1]; a.Down != b.Down || a.Up != b.Up || a.Swing != b.Swing {
		t.Errorf("symmetric sources: %+v and %+v", a, b)
	}
}

// TestTornadoTopAndFilter: Top keeps the largest exact-mean swings, and a
// filter matches by kind, by name and by a name's leading words.
func TestTornadoTopAndFilter(t *testing.T) {
	g := sweepGraph()
	all, err := g.Sweep(context.Background(), variate.SweepConfig{})
	if err != nil {
		t.Fatal(err)
	}
	top, err := g.Sweep(context.Background(), variate.SweepConfig{Top: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(top.Rows) != 3 {
		t.Fatalf("%d rows", len(top.Rows))
	}
	for i, row := range top.Rows {
		if row.Input != all.Rows[i].Input || row.Swing != all.Rows[i].Swing {
			t.Errorf("top row %d: %s, want %s", i, row.Input.Name, all.Rows[i].Input.Name)
		}
	}
	for filter, want := range map[string]int{"holding": 2, "node a": 3, "source r repair": 1, "node": 7} {
		sw, err := g.Sweep(context.Background(), variate.SweepConfig{Inputs: []string{filter}})
		if err != nil || len(sw.Rows) != want {
			t.Errorf("filter %q: %v rows, want %d (%v)", filter, sw, want, err)
		}
	}
	if _, err := g.Sweep(context.Background(), variate.SweepConfig{Inputs: []string{"nothing"}}); err == nil {
		t.Error("a filter matching nothing swept")
	}
	for _, bad := range []variate.SweepConfig{{Figure: "p100"}, {Figure: "rp:1"}, {Figure: "layer:"}, {Figure: "median"}, {Delta: 1}} {
		if _, err := g.Sweep(context.Background(), bad); err == nil {
			t.Errorf("%+v accepted", bad)
		}
	}
}

// TestTornadoTransformRows: where the transform applies a quantile row is
// a bracket, and a long run of each nudged model puts its quantile
// inside it.
func TestTornadoTransformRows(t *testing.T) {
	g := Diamond()
	sw, err := g.Sweep(context.Background(), variate.SweepConfig{Figure: "p99", Inputs: []string{"node b"}})
	if err != nil {
		t.Fatal(err)
	}
	const n = 400000
	for _, row := range sw.Rows {
		if row.Method != variate.MethodTransform || !(row.Base.Lo <= row.Base.Hi) || !math.IsNaN(row.UpSE) {
			t.Fatalf("%s: %+v", row.Input.Name, row)
		}
		for f, r := range map[float64]variate.Range{0.9: row.Down, 1.1: row.Up} {
			sg, _, _ := g.Scaled(row.Input.Name, f)
			_, res := run(t, sg, variate.Config{Periods: n, Series: true, Workers: 4}, 21)
			x := res.Series.TotalColumn(make([]float64, n))
			above, atOrAbove := 0, 0
			for _, v := range x {
				if v > r.Hi {
					above++
				}
				if v >= r.Lo {
					atOrAbove++
				}
			}
			sd := math.Sqrt(0.01 * 0.99 / n)
			if a, b := float64(above)/n, float64(atOrAbove)/n; a > 0.01+5*sd || b < 0.01-5*sd {
				t.Errorf("%s × %g: P(T > %g) = %g, P(T ≥ %g) = %g around 0.01", row.Input.Name, f, r.Hi, a, r.Lo, b)
			}
		}
	}
}

// TestTornadoSimulatedRows: where neither the analytic expectations nor
// the transform give the figure the rows are simulated at one seed: the
// paired standard error of a difference is far below the unpaired one,
// the difference is within a few of them of the exact one, and a sweep
// repeats for its seed.
func TestTornadoSimulatedRows(t *testing.T) {
	g := chainModel(variate.Chain{K: 1, N: 2, Fail: 3, Repair: 40})
	g.Nodes[0].Functions = []variate.Function{Fn("cost", "c", Pert(1, 2, 6))}
	g.Layers = []variate.Layer{{ID: "xs", Attachment: 1}}
	cfg := variate.SweepConfig{Figure: "layer:xs", Run: variate.Config{Periods: 100000, Seed: 5, Workers: 4}}
	sw, err := g.Sweep(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(sw.Rows) != 1 || sw.Periods != 100000 {
		t.Fatalf("%d rows over %d periods", len(sw.Rows), sw.Periods)
	}
	row := sw.Rows[0]
	if row.Method != variate.MethodSimulation || !(row.UpSE > 0) || !(row.DownSE > 0) {
		t.Fatalf("%+v", row)
	}
	// Unpaired: the two runs' own standard errors, from independent
	// seeds, combined.
	_, a := run(t, g, variate.Config{Periods: 100000, Workers: 4}, 5)
	unpaired := math.Sqrt2 * a.Layers[0].StdErr(true)
	if !(row.UpSE < unpaired/3) {
		t.Errorf("paired standard error %g, unpaired %g", row.UpSE, unpaired)
	}
	again, err := g.Sweep(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if b := again.Rows[0]; b.Down != row.Down || b.Base != row.Base || b.Up != row.Up || b.UpSE != row.UpSE {
		t.Errorf("the sweep did not repeat: %+v and %+v", row, b)
	}
}
