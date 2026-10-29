package suite

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// scheduleModel is a Poisson source of rate 2 booking 1 per round, its
// rate following a two-value cycle, with a second source of rate 3 on
// the same cycle when shared.
func scheduleModel(values []float64, share bool) variate.Graph {
	g := variate.Graph{
		Schedules: []variate.Schedule{{ID: "season", Values: values}},
		Sources:   []variate.Source{{ID: "a", Frequency: C(2), To: "n", Schedule: "season"}, {ID: "b", Frequency: C(3), To: "n"}},
		Nodes:     []variate.Node{{ID: "n", Functions: []variate.Function{Fn("one", "c", C(1))}}},
	}
	if share {
		g.Sources[1].Schedule = "season"
	}
	return g
}

// TestScheduleMoments: a cycle of 0.5 and 1.5 has mean 1 and E[S²] =
// 1.25, so on one rate E[T] = 5 and Var[T] = 5 + 4 × 0.25; on both,
// Var[T] = 5 + 25 × 0.25; and P(T = 0) averages over the phases.
func TestScheduleMoments(t *testing.T) {
	for _, tc := range []struct {
		name  string
		share bool
		want  float64
		pzero float64
	}{
		{"one source", false, 5 + 4*0.25, (math.Exp(-1-3) + math.Exp(-3-3)) / 2},
		{"shared", true, 5 + 25*0.25, (math.Exp(-2.5) + math.Exp(-7.5)) / 2},
	} {
		g := scheduleModel([]float64{0.5, 1.5}, tc.share)
		c, r := run(t, g, variate.Config{Periods: 200000, Workers: 4}, 23)
		e, _ := c.Analytic()
		if math.Abs(e.Mean-5) > 1e-12 || math.Abs(e.Var-tc.want) > 1e-9 || math.Abs(e.PZero()-tc.pzero) > 1e-12 {
			t.Errorf("%s: exact mean %g var %g P0 %g, want 5, %g and %g", tc.name, e.Mean, e.Var, e.PZero(), tc.want, tc.pzero)
		}
		se := r.Total.StdErr(false)
		if z := (r.Total.Mean() - 5) / se; math.Abs(z) > 5 {
			t.Errorf("%s: simulated mean %g ± %g", tc.name, r.Total.Mean(), se)
		}
		if v := r.Total.StdDev() * r.Total.StdDev(); math.Abs(v-tc.want) > 0.05*tc.want {
			t.Errorf("%s: simulated variance %g vs exact %g", tc.name, v, tc.want)
		}
	}
}

// TestSchedulePhases: the series shows the cycle, period p using the
// value at p modulo the length: a rate of 2 under 0.5, 1, 1.5 gives
// means of 1, 2 and 3 in turn.
func TestSchedulePhases(t *testing.T) {
	g := scheduleModel([]float64{0.5, 1, 1.5}, false)
	g.Sources = g.Sources[:1]
	const h = 90000
	_, r := run(t, g, variate.Config{Periods: h, Workers: 4, Series: true}, 24)
	var sum [3]float64
	for p := int64(0); p < h; p++ {
		sum[p%3] += r.Series.Total(p)
	}
	for i, want := range []float64{1, 2, 3} {
		mean := sum[i] / (h / 3)
		if math.Abs(mean-want) > 4*math.Sqrt(want/(h/3)) {
			t.Errorf("phase %d: mean %g, want %g", i, mean, want)
		}
	}
}

// TestScheduleTransform: the exact distribution mixes over the phases
// and brackets the simulated curve.
func TestScheduleTransform(t *testing.T) {
	g := scheduleModel([]float64{0.5, 1, 1.5, 1}, true)
	ln, _ := dist.NewLognormal(5, 0.6)
	g.Nodes[0].Functions = []variate.Function{Fn("loss", "c", ln)}
	m, tr := transformOf(t, g, variate.TransformConfig{Tol: 1e-2})
	if !tr.Applicable {
		t.Fatalf("not applicable: %s", tr.Reason)
	}
	if tr.Passes < 2 {
		t.Errorf("the schedule should make the transform mix: %d passes", tr.Passes)
	}
	const h = 40000
	_, r := run(t, g, variate.Config{Periods: h, Series: true}, 25)
	totals := r.Series.TotalColumn(make([]float64, h))
	checkBracketsSeries(t, "total", tr.Total, totals, 100, 4.5)
	checkQuantilesSeries(t, "total", tr.Total, totals, 4.5)
	e, _ := m.Analytic()
	want := math.Exp(-2.5)/4 + math.Exp(-5)/2 + math.Exp(-7.5)/4
	if math.Abs(e.PZero()-want) > 1e-9 || math.Abs(tr.PZero-want) > 1e-6 {
		t.Errorf("P(T = 0): exact %g transform %g, want %g", e.PZero(), tr.PZero, want)
	}
}

// TestScheduleErrors: an unknown schedule, an empty cycle, a negative
// value and a schedule on a non-Poisson law are refused.
func TestScheduleErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		mod  func(*variate.Graph)
		want string
	}{
		{"unknown", func(g *variate.Graph) { g.Sources[0].Schedule = "x" }, "schedule x not found"},
		{"empty", func(g *variate.Graph) { g.Schedules[0].Values = nil }, "needs 1 to"},
		{"negative", func(g *variate.Graph) { g.Schedules[0].Values[0] = -1 }, "finite and not negative"},
		{"duplicate", func(g *variate.Graph) { g.Schedules = append(g.Schedules, g.Schedules[0]) }, "duplicate id"},
		{"every", func(g *variate.Graph) { g.Sources[0].Every = 2 }, "Poisson rate only"},
	} {
		g := scheduleModel([]float64{0.5, 1.5}, false)
		tc.mod(&g)
		_, _, err := g.Compile(variate.Config{})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want %q", tc.name, err, tc.want)
		}
	}
}
