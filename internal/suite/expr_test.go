package suite

import (
	"context"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// jobGraph: 5 jobs a period each draw hours H, gamma of mean 8 and sd
// 2, a wage R, gamma of mean 60 and sd 10, and parts P, gamma of mean
// 300 and sd 100, kept outside the total, and book the bill expr to it.
func jobGraph(expr string) variate.Graph {
	h, _ := dist.NewGammaFromMeanSD(8, 2)
	r, _ := dist.NewGammaFromMeanSD(60, 10)
	p, _ := dist.NewGammaFromMeanSD(300, 100)
	return variate.Graph{ID: "job",
		OutsideTotal: []string{"hours", "wage", "parts"},
		Sources:      []variate.Source{{ID: "s", Frequency: C(5), To: "job"}},
		Nodes: []variate.Node{{ID: "job", Functions: []variate.Function{
			Fn("hours", "hours", h), Fn("wage", "wage", r), Fn("parts", "parts", p),
			{Name: "bill", Category: "bill", Expr: expr},
		}}},
	}
}

// jobExact is the bill's exact mean and variance, a compound Poisson of
// rate 5 of V = H·R + P, or H·62.5 + P with the wage a constant:
// λE[V] and λE[V²] = λ(E[H²]E[R²] + 2E[H]E[R]E[P] + E[P²]).
func jobExact(drawnWage bool) (mean, vr float64) {
	const lam, mh, m2h, mp, m2p = 5.0, 8.0, 68.0, 300.0, 100000.0
	mr, m2r := 62.5, 62.5*62.5
	if drawnWage {
		mr, m2r = 60, 3700
	}
	return lam * (mh*mr + mp), lam * (m2h*m2r + 2*mh*mr*mp + m2p)
}

// TestExprExact: the exact mean and variance of hours × wage + parts are
// the closed forms, the wage drawn or a constant, and the categories
// booked beside it keep their own.
func TestExprExact(t *testing.T) {
	for _, drawn := range []bool{true, false} {
		expr := "hours * wage + parts"
		if !drawn {
			expr = "hours * 62.5 + parts"
		}
		m, _, err := jobGraph(expr).Compile(variate.Config{})
		if err != nil {
			t.Fatal(err)
		}
		e, err := m.Analytic()
		if err != nil {
			t.Fatal(err)
		}
		mean, vr := jobExact(drawn)
		if math.Abs(e.Mean-mean) > 1e-12*mean || math.Abs(e.Var-vr) > 1e-12*vr {
			t.Errorf("%s: mean %v var %v, want %v and %v", expr, e.Mean, e.Var, mean, vr)
		}
		bill := indexOf(m.Categories(), "bill")
		if math.Abs(e.MeanByCategory[bill]-mean) > 1e-12*mean || math.Abs(e.VarByCategory[bill]-vr) > 1e-12*vr {
			t.Errorf("%s: bill %v %v", expr, e.MeanByCategory[bill], e.VarByCategory[bill])
		}
		if h := e.MeanByCategory[indexOf(m.Categories(), "hours")]; math.Abs(h-40) > 1e-12 {
			t.Errorf("%s: hours %v, want 40", expr, h)
		}
	}
}

// TestExprSimulated: over 200,000 periods the simulated mean and
// standard deviation are within their errors of the exact ones.
func TestExprSimulated(t *testing.T) {
	const periods = 200000
	for _, drawn := range []bool{true, false} {
		expr := "hours * wage + parts"
		if !drawn {
			expr = "(hours) * (60 + 2.5) + parts"
		}
		_, _, s := runSummary(t, jobGraph(expr), periods, 17)
		mean, vr := jobExact(drawn)
		sd := math.Sqrt(vr)
		if math.Abs(s.Total.Mean-mean) > 4*s.Total.StdErr || math.Abs(s.Total.StdDev-sd) > 0.01*sd {
			t.Errorf("%s: %v ± %v sd %v, want %v and %v", expr, s.Total.Mean, s.Total.StdErr, s.Total.StdDev, mean, sd)
		}
	}
}

// TestExprSingle: an expression that is one name books that value to
// the bit, and adding it changes nothing else the run reports.
func TestExprSingle(t *testing.T) {
	plain := jobGraph("")
	plain.Nodes[0].Functions = plain.Nodes[0].Functions[:3]
	plain.OutsideTotal = []string{"wage", "parts"}
	_, _, want := runSummary(t, plain, 20000, 9)
	copied := jobGraph("hours")
	copied.OutsideTotal = []string{"wage", "parts", "bill"}
	_, _, got := runSummary(t, copied, 20000, 9)
	if got.Total != want.Total || got.ByCategory["hours"] != want.ByCategory["hours"] {
		t.Errorf("total %v, without %v", got.Total, want.Total)
	}
	bill, hours := got.ByCategory["bill"], got.ByCategory["hours"]
	bill.Share, hours.Share = 0, 0 // bill is outside the total, hours the whole of it
	if bill != hours {
		t.Errorf("bill %v, hours %v", got.ByCategory["bill"], got.ByCategory["hours"])
	}
}

// TestExprNested: an expression may name another, and a constant
// factor on it scales it; the mean is the closed form.
func TestExprNested(t *testing.T) {
	g := jobGraph("labour + parts")
	g.Factors = []variate.Factor{{ID: "two", Model: C(2)}}
	g.OutsideTotal = append(g.OutsideTotal, "labour")
	g.Nodes[0].Functions = append(g.Nodes[0].Functions, variate.Function{Name: "labour", Category: "labour", Expr: "hours * wage"})
	g.Nodes[0].Functions[3].Factor = "two"
	m, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	mean, vr := jobExact(true)
	if math.Abs(e.Mean-2*mean) > 1e-12*mean || math.Abs(e.Var-4*vr) > 1e-12*vr {
		t.Errorf("mean %v var %v, want %v and %v", e.Mean, e.Var, 2*mean, 4*vr)
	}
}

// TestExprFactor: a drawn factor on an expression scales its value, the
// exact figures conditioned on it: 5·E[G]·E[V] and
// 5·E[G²]·E[V²] + 25·Var(G)·E[V]².
func TestExprFactor(t *testing.T) {
	g := jobGraph("hours * wage + parts")
	ln, _ := dist.NewLognormalFromMeanSD(1.1, 0.4)
	g.Factors = []variate.Factor{{ID: "inflation", Model: ln}}
	g.Nodes[0].Functions[3].Factor = "inflation"
	m, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	mean, vr := jobExact(true)
	ev, ev2 := mean/5, vr/5
	g1, g2 := lognormalMoment(1.1, 0.4, 1), lognormalMoment(1.1, 0.4, 2)
	wantMean, wantVar := 5*g1*ev, 5*g2*ev2+25*(g2-g1*g1)*ev*ev
	if math.Abs(e.Mean-wantMean) > 1e-9*wantMean || math.Abs(e.Var-wantVar) > 1e-9*wantVar {
		t.Errorf("mean %v var %v, want %v and %v", e.Mean, e.Var, wantMean, wantVar)
	}
	_, _, s := runSummary(t, g, 200000, 4)
	if math.Abs(s.Total.Mean-wantMean) > 4*s.Total.StdErr || math.Abs(s.Total.StdDev-math.Sqrt(wantVar)) > 0.02*math.Sqrt(wantVar) {
		t.Errorf("simulated %v ± %v sd %v", s.Total.Mean, s.Total.StdErr, s.Total.StdDev)
	}
}

// TestExprRefusals: an expression is refused that squares a draw, names
// itself round a cycle, names what is not a drawn on-entry function of
// its node, may be below 0, or is malformed; and where it cannot stand.
func TestExprRefusals(t *testing.T) {
	for expr, want := range map[string]string{
		"hours * hours":          "multiplied by itself",
		"hours * (wage + hours)": "multiplied by itself",
		"bill + 1":               "cycle",
		"crew * 2":               "names crew, not a function",
		"parts - 500":            "may be below 0",
		"hours * ":               "incomplete",
		"hours wage":             "two operands",
		"(hours + 1":             "incomplete",
		"hours + 1)":             "unmatched )",
		"hours / 2":              "unexpected",
		strings.Repeat("(", 17) + "1" + strings.Repeat(")", 17): "deeper than 16",
	} {
		_, _, err := jobGraph(expr).Compile(variate.Config{})
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%q: %v, want %q", expr, err, want)
		}
	}
	both := jobGraph("hours")
	both.Nodes[0].Functions[3].Model = C(1)
	perTime := jobGraph("hours")
	perTime.Nodes[0].Functions[3].Basis = variate.PerTime
	perTime.Nodes[0].Next = "end"
	perTime.Nodes = append(perTime.Nodes, variate.Node{ID: "end"})
	thresh := jobGraph("hours")
	thresh.Nodes[0].Threshold = &variate.Threshold{Function: "bill", Over: 3, Above: "end", Below: "end"}
	thresh.Nodes = append(thresh.Nodes, variate.Node{ID: "end"})
	tested := jobGraph("hours")
	tested.Nodes[0].Threshold = &variate.Threshold{Function: "hours", Over: 3, Above: "end", Below: "end"}
	tested.Nodes = append(tested.Nodes, variate.Node{ID: "end"})
	trig := jobGraph("hours * wage")
	trig.Nodes = append(trig.Nodes, variate.Node{ID: "audit", Trigger: &variate.Trigger{Over: 1, Inputs: []variate.TriggerInput{{Node: "job"}}}})
	for name, c := range map[string]struct {
		g    variate.Graph
		want string
	}{
		"both":     {both, "both a distribution and an expression"},
		"per-time": {perTime, "a per-time function is a rate"},
		"tests":    {thresh, "tests a drawn value"},
		"tested":   {tested, "the value the threshold tests"},
		"trigger":  {trig, "trigger input job.bill is an expression"},
	} {
		if _, _, err := c.g.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want %q", name, err, c.want)
		}
	}
}

// TestExprTransform: the exact distribution is not computed for a model
// with an expression, and says why.
func TestExprTransform(t *testing.T) {
	_, tr := transformOf(t, jobGraph("hours * wage + parts"), variate.TransformConfig{})
	if tr.Applicable || !strings.Contains(tr.Reason, "an expression") {
		t.Errorf("applicable %v: %s", tr.Applicable, tr.Reason)
	}
}

// TestExprPeriod: a wage drawn once a period is shared by the period's
// jobs, adding λ²E[H]²Var(R) to the variance, exact and simulated.
func TestExprPeriod(t *testing.T) {
	g := jobGraph("hours * wage + parts")
	g.Nodes[0].Functions[1].Scope = variate.PerPeriod
	m, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	mean, vr := jobExact(true)
	vr += 25 * 64 * 100
	if math.Abs(e.Mean-mean) > 1e-12*mean || math.Abs(e.Var-vr) > 1e-12*vr {
		t.Errorf("mean %v var %v, want %v and %v", e.Mean, e.Var, mean, vr)
	}
	_, _, s := runSummary(t, g, 200000, 8)
	if math.Abs(s.Total.Mean-mean) > 4*s.Total.StdErr || math.Abs(s.Total.StdDev-math.Sqrt(vr)) > 0.01*math.Sqrt(vr) {
		t.Errorf("simulated %v ± %v sd %v, want %v and %v", s.Total.Mean, s.Total.StdErr, s.Total.StdDev, mean, math.Sqrt(vr))
	}
}

// TestExprStarted: a started run's exact mean per period is the steady
// state's, every period of an empty start.
func TestExprStarted(t *testing.T) {
	g := jobGraph("hours * wage + parts")
	g.Start = variate.Start{Horizon: 3}
	m, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	want, _ := jobExact(true)
	for k, row := range e.ByPeriod {
		if math.Abs(row[0]-want) > 1e-12*want {
			t.Errorf("period %d: %v, want %v", k, row[0], want)
		}
	}
	if math.Abs(e.Mean-3*want) > 1e-12*want {
		t.Errorf("horizon %v, want %v", e.Mean, 3*want)
	}
}

// TestExprSweep: an expression is not an input of the sweep, its
// operands are, and the sweep runs.
func TestExprSweep(t *testing.T) {
	g := jobGraph("hours * wage + parts")
	for _, in := range g.Inputs() {
		if strings.Contains(in.Name, "bill") {
			t.Errorf("input %s", in.Name)
		}
	}
	if len(g.Inputs()) != 4 {
		t.Errorf("%d inputs, want the frequency and 3 functions", len(g.Inputs()))
	}
	if _, err := g.Sweep(context.Background(), variate.SweepConfig{Screen: 0.1}); err != nil {
		t.Fatal(err)
	}
}
