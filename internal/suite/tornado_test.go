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

// TestTornadoMeanRowsExact: on the corpus, screened, every mean row is
// exact and is the analytic mean of the scaled model, the base row the
// model's own; a model with many inputs has every 15th row checked so.
func TestTornadoMeanRowsExact(t *testing.T) {
	for _, cm := range corpus(t) {
		base, err := mustCompile(t, cm.g).Analytic()
		if err != nil {
			t.Fatalf("%s: %v", cm.name, err)
		}
		sw, err := cm.g.Sweep(context.Background(), variate.SweepConfig{Screen: 0.1})
		if err != nil {
			t.Fatalf("%s: %v", cm.name, err)
		}
		if len(sw.Rows) != len(cm.g.Inputs()) {
			t.Errorf("%s: %d rows for %d inputs", cm.name, len(sw.Rows), len(cm.g.Inputs()))
		}
		for i, row := range sw.Rows {
			if row.Method != variate.MethodExact || row.Base.Lo != base.Mean || row.LowSE != 0 {
				t.Fatalf("%s %s: %s row, base %g, want exact %g", cm.name, row.Input.Name, row.Method, row.Base.Lo, base.Mean)
			}
			if i > 0 && row.Swing > sw.Rows[i-1].Swing {
				t.Errorf("%s: rows not by swing", cm.name)
			}
			if len(sw.Rows) > 40 && i%15 != 0 {
				continue
			}
			for f, got := range map[float64]float64{0.9: row.AtLow.Lo, 1.1: row.AtHigh.Lo} {
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
	sw, err := g.Sweep(context.Background(), variate.SweepConfig{Screen: 0.2, Inputs: []string{"frequency"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(sw.Rows) != 2 {
		t.Fatalf("%d rows", len(sw.Rows))
	}
	for _, row := range sw.Rows {
		k := slices.Index([]string{"source s frequency", "source t frequency"}, row.Input.Name)
		share := 0.2 * e.MeanBySource[k]
		if math.Abs(row.AtHigh.Lo-row.Base.Lo-share) > 1e-12*share || math.Abs(row.Base.Lo-row.AtLow.Lo-share) > 1e-12*share {
			t.Errorf("%s: %g %g %g, want ±%g", row.Input.Name, row.AtLow.Lo, row.Base.Lo, row.AtHigh.Lo, share)
		}
	}
	if a, b := sw.Rows[0], sw.Rows[1]; a.AtLow != b.AtLow || a.AtHigh != b.AtHigh || a.Swing != b.Swing {
		t.Errorf("symmetric sources: %+v and %+v", a, b)
	}
}

// TestTornadoTopAndFilter: Top keeps the largest exact-mean swings, a
// filter matches by kind, by name and by a name's leading words, and a
// model with no ranges is not swept but may be screened.
func TestTornadoTopAndFilter(t *testing.T) {
	g := sweepGraph()
	all, err := g.Sweep(context.Background(), variate.SweepConfig{Screen: 0.1})
	if err != nil {
		t.Fatal(err)
	}
	top, err := g.Sweep(context.Background(), variate.SweepConfig{Screen: 0.1, Top: 3})
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
		sw, err := g.Sweep(context.Background(), variate.SweepConfig{Screen: 0.1, Inputs: []string{filter}})
		if err != nil || len(sw.Rows) != want {
			t.Errorf("filter %q: %v rows, want %d (%v)", filter, sw, want, err)
		}
	}
	if _, err := g.Sweep(context.Background(), variate.SweepConfig{Screen: 0.1, Inputs: []string{"nothing"}}); err == nil {
		t.Error("a filter matching nothing swept")
	}
	if _, err := g.Sweep(context.Background(), variate.SweepConfig{}); err == nil || !strings.Contains(err.Error(), "no input has a range") {
		t.Errorf("a model without ranges swept: %v", err)
	}
	for _, bad := range []variate.SweepConfig{{Screen: 0.1, Figure: "p100"}, {Screen: 0.1, Figure: "rp:1"}, {Screen: 0.1, Figure: "layer:"}, {Screen: 0.1, Figure: "median"}, {Screen: 1}, {Screen: -0.1}} {
		if _, err := g.Sweep(context.Background(), bad); err == nil {
			t.Errorf("%+v accepted", bad)
		}
	}
}

// TestTornadoTransformRows: where the transform applies a quantile row is
// a bracket, and a long run of each moved model puts its quantile
// inside it.
func TestTornadoTransformRows(t *testing.T) {
	g := Diamond()
	sw, err := g.Sweep(context.Background(), variate.SweepConfig{Figure: "p99", Screen: 0.1, Inputs: []string{"node b"}})
	if err != nil {
		t.Fatal(err)
	}
	const n = 400000
	for _, row := range sw.Rows {
		if row.Method != variate.MethodTransform || !(row.Base.Lo <= row.Base.Hi) || !math.IsNaN(row.HighSE) {
			t.Fatalf("%s: %+v", row.Input.Name, row)
		}
		for f, r := range map[float64]variate.Range{0.9: row.AtLow, 1.1: row.AtHigh} {
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
	cfg := variate.SweepConfig{Figure: "layer:xs", Screen: 0.1, Run: variate.Config{Periods: 100000, Seed: 5, Workers: 4}}
	sw, err := g.Sweep(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(sw.Rows) != 1 || sw.Periods != 100000 {
		t.Fatalf("%d rows over %d periods", len(sw.Rows), sw.Periods)
	}
	row := sw.Rows[0]
	if row.Method != variate.MethodSimulation || !(row.HighSE > 0) || !(row.LowSE > 0) {
		t.Fatalf("%+v", row)
	}
	// Unpaired: the two runs' own standard errors, from independent
	// seeds, combined.
	_, a := run(t, g, variate.Config{Periods: 100000, Workers: 4}, 5)
	unpaired := math.Sqrt2 * a.Layers[0].StdErr(true)
	if !(row.HighSE < unpaired/3) {
		t.Errorf("paired standard error %g, unpaired %g", row.HighSE, unpaired)
	}
	again, err := g.Sweep(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if b := again.Rows[0]; b.AtLow != row.AtLow || b.Base != row.Base || b.AtHigh != row.AtHigh || b.HighSE != row.HighSE {
		t.Errorf("the sweep did not repeat: %+v and %+v", row, b)
	}
}

// TestTornadoRanges: only the inputs with a range are swept, each to the
// two ends of it. A range in the input's units is read as its mean at
// each end, so the factors are the ends over the mean, and a fixed
// probability's range is the probability itself at each end; a range
// given as factors is used as it is. A range must be positive, ordered
// and hold the mean.
func TestTornadoRanges(t *testing.T) {
	g := sweepGraph()
	freq := g.Sources[0].Frequency.Mean()
	g.Sources[0].FrequencyRange = &variate.InputRange{Low: 1.5, High: 4}
	g.Nodes[0].Outcomes[0].PRange = &variate.InputRange{Low: 0.2, High: 0.5}
	g.Nodes[3].Functions[0].Range = &variate.InputRange{Low: 0.5, High: 2, Factors: true}
	sw, err := g.Sweep(context.Background(), variate.SweepConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if sw.Screen != 0 || len(sw.Rows) != 3 {
		t.Fatalf("screen %g, %d rows", sw.Screen, len(sw.Rows))
	}
	want := map[string][2]float64{
		"source s frequency": {1.5 / freq, 4 / freq},
		"node a outcome b p": {0.2 / 0.3, 0.5 / 0.3},
		"node d function fd": {0.5, 2},
	}
	for _, row := range sw.Rows {
		f, ok := want[row.Input.Name]
		if !ok || math.Abs(row.Low-f[0]) > 1e-15 || math.Abs(row.High-f[1]) > 1e-15 {
			t.Errorf("%s: factors %g, %g, want %v", row.Input.Name, row.Low, row.High, f)
			continue
		}
		for i, got := range []float64{row.AtLow.Lo, row.AtHigh.Lo} {
			sg, _, _ := g.Scaled(row.Input.Name, f[i])
			e, err := mustCompile(t, sg).Analytic()
			if err != nil || e.Mean != got {
				t.Errorf("%s at end %d: %g, analytic %g (%v)", row.Input.Name, i, got, e.Mean, err)
			}
		}
	}
	// The probability is the range's end: 0.3 → 0.2 and 0.5, the other
	// outcome taking up the rest.
	sg, _, _ := g.Scaled("node a outcome b p", 0.5/0.3)
	if p := sg.Nodes[0].Outcomes[0].P.Mean(); math.Abs(p-0.5) > 1e-15 {
		t.Errorf("probability at the high end: %g", p)
	}
	for name, r := range map[string]variate.InputRange{
		"reversed":         {Low: 4, High: 1.5},
		"zero":             {Low: 0, High: 4},
		"misses the mean":  {Low: 3, High: 4},
		"factors miss one": {Low: 1.1, High: 2, Factors: true},
		"infinite":         {Low: 1, High: math.Inf(1)},
	} {
		bad := g
		bad.Sources = slices.Clone(g.Sources)
		bad.Sources[0].FrequencyRange = &r
		if _, err := bad.Sweep(context.Background(), variate.SweepConfig{}); err == nil || !strings.Contains(err.Error(), "source s frequency range") {
			t.Errorf("%s: %v", name, err)
		}
	}
	zero := g
	zero.Nodes = slices.Clone(g.Nodes)
	zero.Nodes[3].Functions = []variate.Function{{Name: "fd", Category: "parts", Model: C(0), Range: &variate.InputRange{Low: 1, High: 2}}}
	if _, err := zero.Sweep(context.Background(), variate.SweepConfig{Inputs: []string{"node d"}}); err == nil || !strings.Contains(err.Error(), "as factors") {
		t.Errorf("a range in units on a zero input: %v", err)
	}
	// A screen moves every input, ranges or none.
	screen, err := g.Sweep(context.Background(), variate.SweepConfig{Screen: 0.1})
	if err != nil || len(screen.Rows) != len(g.Inputs()) || screen.Rows[0].Low != 0.9 {
		t.Errorf("screen: %v rows, %v", screen, err)
	}
}
