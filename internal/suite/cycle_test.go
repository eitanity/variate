package suite

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
)

// opposed is two unit sources on opposite schedules, [2, 0] and [0, 2]:
// every period the total is Poisson(2), the schedules never both high.
func opposed() variate.Graph {
	return variate.Graph{
		Schedules: []variate.Schedule{{ID: "a", Values: []float64{2, 0}}, {ID: "b", Values: []float64{0, 2}}},
		Sources:   []variate.Source{{ID: "x", Frequency: C(1), To: "n", Schedule: "a"}, {ID: "y", Frequency: C(1), To: "n", Schedule: "b"}},
		Nodes:     []variate.Node{{ID: "n", Functions: []variate.Function{Fn("one", "c", C(1))}}},
	}
}

// TestJointSchedules: the exact variance, P(T = 0) and the transform's
// bracket are those of Poisson(2), not of two independent multipliers.
func TestJointSchedules(t *testing.T) {
	c, _, err := opposed().Compile(variate.Config{Transform: true})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := c.Analytic()
	if math.Abs(e.Var-2) > 1e-12 || math.Abs(e.PZero()-math.Exp(-2)) > 1e-12 {
		t.Errorf("variance %g and P(T = 0) %g, want 2 and %g", e.Var, e.PZero(), math.Exp(-2))
	}
	tr := c.Transformed()
	if tr == nil || !tr.Applicable {
		t.Fatalf("no transform: %+v", tr)
	}
	surv := 1.0
	for k := range 6 {
		i := int(math.Round(float64(k) / tr.Total.H))
		if lo, hi := tr.Total.Lower[i], tr.Total.Upper[i]; surv < lo-1e-9 || surv > hi+1e-9 {
			t.Errorf("P(T ≥ %d) = %g outside the bracket [%g, %g]", k, surv, lo, hi)
		}
		surv -= math.Exp(-2) * math.Pow(2, float64(k)) / math.Gamma(float64(k)+1)
	}
}

// TestJointCycle: for schedules whose lengths share a factor, or do not,
// the whole cycle's variance and P(T = 0) are those of a phase drawn
// uniformly from the combined cycle.
func TestJointCycle(t *testing.T) {
	for _, pair := range [][2][]float64{
		{{3, 0, 2, 0}, {0, 1, 0, 2, 0, 4}},
		{{1, 2, 3}, {4, 0, 1, 2}},
		{{0, 1, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}},
	} {
		g := opposed()
		g.Schedules[0].Values, g.Schedules[1].Values = pair[0], pair[1]
		lens := [2]int{len(pair[0]), len(pair[1])}
		g.Nodes[0].Functions = []variate.Function{Fn("loss", "c", Pert(1, 3, 10))}
		c, _, err := g.Compile(variate.Config{})
		if err != nil {
			t.Fatal(err)
		}
		whole, _ := c.Analytic()
		n, _ := g.Cycle()
		var varSum, meanSum, meanSq, p0 float64
		for i := 1; i <= n; i++ {
			p, _ := g.Phase(i)
			pc, _, err := p.Compile(variate.Config{})
			if err != nil {
				t.Fatal(err)
			}
			e, _ := pc.Analytic()
			varSum += e.Var
			meanSum += e.Mean
			meanSq += e.Mean * e.Mean
			p0 += e.PZero() / float64(n)
		}
		mean := meanSum / float64(n)
		if total := varSum/float64(n) + meanSq/float64(n) - mean*mean; math.Abs(total-whole.Var) > 1e-9*whole.Var {
			t.Errorf("lengths %v: phases give variance %g, whole cycle %g", lens, total, whole.Var)
		}
		if math.Abs(p0-whole.PZero()) > 1e-12 {
			t.Errorf("lengths %v: phases give P(T = 0) %g, whole cycle %g", lens, p0, whole.PZero())
		}
	}
}

// TestJointCycleTooLong: schedules of 2048 and 2047 values make a cycle
// of over four million periods, too long to walk; the variance is still
// exact, P(T = 0) is NaN and the transform says why it does not apply.
func TestJointCycleTooLong(t *testing.T) {
	g := opposed()
	for s, n := range []int{2048, 2047} {
		v := make([]float64, n)
		for i := range v {
			v[i] = float64(i%3) * float64(s+1)
		}
		g.Schedules[s].Values = v
	}
	c, _, err := g.Compile(variate.Config{Transform: true})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := c.Analytic()
	if math.IsNaN(e.Var) || !math.IsNaN(e.PZero()) {
		t.Errorf("variance %g and P(T = 0) %g, want a variance and NaN", e.Var, e.PZero())
	}
	if tr := c.Transformed(); tr == nil || tr.Applicable || !strings.Contains(tr.Reason, "combined cycle") {
		t.Errorf("transform %+v, want refused for the combined cycle", tr)
	}
}
