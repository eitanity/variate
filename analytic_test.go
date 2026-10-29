package variate

import (
	"math"
	"testing"

	"code.eitanity.com/variate/dist"
)

func TestAnalyticDiamond(t *testing.T) {
	m, _ := mustCompile(t, diamond(), Config{})
	e, err := m.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	lam := (1 + 4*2 + 4) / 6.0 // PERT(1,2,4) mean
	// E[T] = λ·(0.3·E[fb] + 10) with E[fb] = 2.
	if !near(e.Mean, lam*(0.3*2+10), 1e-12) {
		t.Errorf("mean %g", e.Mean)
	}
	if !near(e.EntriesPer[1], 0.3*lam, 1e-12) || !near(e.EntriesPer[3], lam, 1e-12) {
		t.Errorf("entries %v", e.EntriesPer)
	}
	if !near(e.OutcomeRate[0], 0.7*lam, 1e-12) || !near(e.OutcomeRate[1], 0.3*lam, 1e-12) {
		t.Errorf("outcome rates %v", e.OutcomeRate)
	}
	if e.Terminal[0][3] != 1 || e.Terminal[0][0] != 0 {
		t.Errorf("terminal %v", e.Terminal[0])
	}
	if !near(e.MeanByCategory[0], lam*0.6, 1e-12) || !near(e.MeanByCategory[1], lam*10, 1e-12) {
		t.Errorf("by category %v", e.MeanByCategory)
	}
	if !near(e.Contrib[1], lam*0.6, 1e-12) || !near(e.Contrib[3], lam*10, 1e-12) || e.Contrib[0] != 0 {
		t.Errorf("contrib %v", e.Contrib)
	}
	if !near(e.StepsPerPeriod, lam*(1+0.3+0.7+1), 1e-12) {
		t.Errorf("steps %g", e.StepsPerPeriod)
	}
	// Draws: one per function entry: node b has one function, node d one.
	if !near(e.DrawsMean, lam*(0.3+1), 1e-12) {
		t.Errorf("draws mean %g", e.DrawsMean)
	}
	// Var[T] with only λ period-scoped: Σ E[λ]·M2 + Var[λ]·M1².
	fb := pert(1, 2, 3)
	efb2 := fb.Var() + 4
	m1 := 0.3*2 + 10
	m2 := 0.3*(efb2+2*2*10+100) + 0.7*100
	varLam := pert(1, 2, 4).Var()
	want := lam*m2 + varLam*m1*m1
	if !near(e.Var, want, 1e-12) {
		t.Errorf("var %.15g want %.15g", e.Var, want)
	}
}

// TestAnalyticPeriodProbability: with λ constant and a period-scoped
// two-way probability p, T = λ·[p·X1 + (1−p)·X2] in expectation per period,
// so Var[T] = λ·E[C²] + λ²·Var[p]·(μ1 − μ2)².
func TestAnalyticPeriodProbability(t *testing.T) {
	g := Graph{
		Sources: []Source{{ID: "s", Frequency: c(3), To: "a"}},
		Nodes: []Node{
			{ID: "a", Outcomes: []Outcome{{To: "x", P: beta(2, 5)}, {To: "y"}}},
			{ID: "x", Functions: []Function{fn("fx", "k", c(10))}},
			{ID: "y", Functions: []Function{fn("fy", "k", c(4))}},
		},
	}
	m, _ := mustCompile(t, g, Config{})
	e, err := m.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	p := beta(2, 5)
	mu := p.Mean()
	ec2 := mu*100 + (1-mu)*16
	want := 3*ec2 + 9*p.Var()*36
	if !near(e.Var, want, 1e-12) {
		t.Errorf("var %.15g want %.15g", e.Var, want)
	}
	if !near(e.Mean, 3*(mu*10+(1-mu)*4), 1e-12) {
		t.Errorf("mean %g", e.Mean)
	}
}

// TestAnalyticSequentialMatchesTwoWay: a sequential node with one
// uncertain q and a remainder is the two-way node.
func TestAnalyticSequentialMatchesTwoWay(t *testing.T) {
	base := func(mode Mode) Graph {
		return Graph{
			Sources: []Source{{ID: "s", Frequency: pert(1, 2, 3), To: "a"}},
			Nodes: []Node{
				{ID: "a", Mode: mode, Outcomes: []Outcome{{To: "x", P: beta(2, 5)}, {To: "y"}}},
				{ID: "x", Functions: []Function{fn("fx", "k", pert(1, 5, 20))}},
				{ID: "y", Functions: []Function{fn("fy", "k", c(4))}},
			},
		}
	}
	m1, _ := mustCompile(t, base(Absolute), Config{})
	m2, _ := mustCompile(t, base(Sequential), Config{})
	e1, _ := m1.Analytic()
	e2, _ := m2.Analytic()
	if !near(e1.Mean, e2.Mean, 1e-14) || !near(e1.Var, e2.Var, 1e-12) {
		t.Errorf("absolute %g %g sequential %g %g", e1.Mean, e1.Var, e2.Mean, e2.Var)
	}
}

func TestAnalyticDelaysAndInfinite(t *testing.T) {
	g := diamond()
	g.Nodes[0].Outcomes[0].Holding = c(1)
	m, _ := mustCompile(t, g, Config{})
	e, _ := m.Analytic()
	if !math.IsNaN(e.Var) || !near(e.Mean, (13.0/6)*(0.6+10), 1e-12) {
		t.Errorf("with delays: var %g mean %g", e.Var, e.Mean)
	}
	g = diamond()
	pa, _ := dist.NewPareto(1, 1.5)
	g.Nodes[3].Functions[0].Model = pa
	m, _ = mustCompile(t, g, Config{})
	e, _ = m.Analytic()
	if !math.IsInf(e.Var, 1) || math.IsInf(e.Mean, 0) {
		t.Errorf("infinite variance: var %g mean %g", e.Var, e.Mean)
	}
}

func near(a, b, rel float64) bool { return math.Abs(a-b) <= rel*math.Max(math.Abs(b), 1e-300) }
