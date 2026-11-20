package suite

import (
	"context"
	"fmt"
	"math"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
	"code.eitanity.com/variate/dist"
)

// heldPath: 8 rounds a period book X, gamma of mean 3 and sd 1, at a,
// and a rate of 4 a period, category lost, over the holding τ to b,
// where Y, gamma of mean 5 and sd 2, is booked.
func heldPath(tau dist.Distribution) variate.Graph {
	x, _ := dist.NewGammaFromMeanSD(3, 1)
	y, _ := dist.NewGammaFromMeanSD(5, 2)
	r := Fn("r", "lost", C(4))
	r.Basis = variate.PerTime
	return variate.Graph{ID: "held",
		Sources: []variate.Source{{ID: "s", Frequency: C(8), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{Fn("x", "x", x), r}, Next: "b", NextHolding: tau},
			{ID: "b", Functions: []variate.Function{Fn("y", "y", y)}},
		},
	}
}

// heldConstExact is heldPath's variance with a constant holding c: Q is
// X² + Y² + 16·G₂(c) + 2XY·K(c) + 8X·H(c) + 8Y·H(c), K(c) = (1 − c)⁺,
// H(c) = ∫₀^c K and G₂(c) = ∫∫_{[0,c]²} K, times λ = 8.
func heldConstExact(c float64) float64 {
	k := math.Max(0, 1-c)
	u := math.Min(c, 1)
	hc := u - u*u/2
	g2 := c*c - c*c*c/3
	if c > 1 {
		g2 = c - 1.0/3
	}
	return 8 * (10 + 29 + 16*g2 + 2*3*5*k + 8*3*hc + 8*5*hc)
}

// TestDelayConstant: with a constant holding the variance is the closed
// form, on the lag grid's points or between them, and past a period.
func TestDelayConstant(t *testing.T) {
	for _, c := range []float64{0.3, 0.5, 0.71234, 1.5} {
		_, e := depExact(t, heldPath(C(c)))
		if want := heldConstExact(c); !within(e.Var, want, 1e-9) {
			t.Errorf("holding %v: var %v, want %v", c, e.Var, want)
		}
	}
}

// TestDelayZero: a holding of 0 books as no holding does: the variance
// of the model without it.
func TestDelayZero(t *testing.T) {
	_, held := depExact(t, heldPath(C(0)))
	g := heldPath(C(0))
	g.Nodes[0].NextHolding = nil
	g.Nodes[0].Functions = g.Nodes[0].Functions[:1]
	_, none := depExact(t, g)
	if !within(held.Var, none.Var, 1e-12) {
		t.Errorf("holding 0: var %v, without %v", held.Var, none.Var)
	}
}

// TestDelaySimulated: with a lognormal holding of median 2 days, 3
// months or a year in a year's period, the exact standard deviation of
// the total, each category, the source and each node is within its
// error of the simulated over 2^20 periods.
func TestDelaySimulated(t *testing.T) {
	for _, med := range []float64{2.0 / 365, 0.25, 1} {
		tau, _ := dist.NewLognormalFromMedianP90(med, 3*med)
		g := heldPath(tau)
		m, e := depExact(t, g)
		_, _, s := runSummary(t, g, 1<<20, 9)
		check := func(what string, vr float64, st variate.Stat) {
			sd := math.Sqrt(vr)
			if se := st.StdDev / math.Sqrt(2*float64(st.N)); !(math.Abs(st.StdDev-sd) <= 4*se) {
				t.Errorf("median %v, %s: exact sd %v, simulated %v ± %v", med, what, sd, st.StdDev, se)
			}
		}
		check("total", e.Var, s.Total)
		for _, c := range []string{"x", "y", "lost"} {
			check("category "+c, e.VarByCategory[indexOf(m.Categories(), c)], s.ByCategory[c])
		}
		check("source", e.VarBySource[0], s.BySource["s"])
		if math.IsNaN(e.ContribVar[0]) || math.IsNaN(e.ContribVar[1]) {
			t.Errorf("median %v: node variances %v", med, e.ContribVar)
		}
	}
}

// TestDelayTotals: a named total's variance with holding times is its
// categories', the lost production and X, against the simulation.
func TestDelayTotals(t *testing.T) {
	tau, _ := dist.NewLognormalFromMedianP90(0.25, 0.75)
	g := heldPath(tau)
	g.Totals = []variate.Total{{ID: "net", Categories: []string{"x"}, Less: []string{"lost"}}}
	_, e := depExact(t, g)
	_, _, s := runSummary(t, g, 1<<20, 4)
	sd, st := math.Sqrt(e.VarByTotal[0]), s.Totals["net"]
	if se := st.StdDev / math.Sqrt(2*float64(st.N)); !(math.Abs(st.StdDev-sd) <= 4*se) {
		t.Errorf("net: exact sd %v, simulated %v ± %v", sd, st.StdDev, se)
	}
}

// TestDelayDependence: values and a holding that move together at rank
// 0.7: the exact standard deviation, conditioned on the group's normal,
// is within its error of the simulated, at holdings of 2 days and of 3
// months.
func TestDelayDependence(t *testing.T) {
	for _, med := range []float64{2.0 / 365, 0.25} {
		tau, _ := dist.NewLognormalFromMedianP90(med, 3*med)
		g := heldPath(tau)
		g.Dependence = []variate.Dependence{{Members: []string{"a.x", "a->b", "b.y"}, Rank: 0.7}}
		_, e := depExact(t, g)
		g.Dependence = nil
		_, base := depExact(t, g)
		g.Dependence = []variate.Dependence{{Members: []string{"a.x", "a->b", "b.y"}, Rank: 0.7}}
		_, _, s := runSummary(t, g, 1<<20, 3)
		sd, se := math.Sqrt(e.Var), s.Total.StdDev/math.Sqrt(2*float64(s.Total.N))
		if !(math.Abs(s.Total.StdDev-sd) <= 4*se) || e.Mean != base.Mean {
			t.Errorf("median %v: exact sd %v (mean %v, apart %v), simulated %v ± %v", med, sd, e.Mean, base.Mean, s.Total.StdDev, se)
		}
	}
}

// TestDelayNearLockstep: values and a holding that move together at rank
// 0.9, 0.95 and 0.99, where a booking's figures given the group's normal
// bend where its holding crosses a period; at holdings of 2 days (to
// rank 0.95), 3 months and a year, with a drawn rate at 3 months, whose
// lag profiles bend at every lag, and on a path of two of the group's
// holdings, the exact variance is within four errors of 2^21 simulated
// periods by total, category and source.
func TestDelayNearLockstep(t *testing.T) {
	members := []string{"a.x", "a->b", "b.y"}
	cases := map[string]variate.Graph{}
	for _, med := range []float64{2.0 / 365, 0.25, 1} {
		for _, rank := range []float64{0.9, 0.95, 0.99} {
			if med < 0.25 && rank > 0.95 {
				continue
			}
			tau, _ := dist.NewLognormalFromMedianP90(med, 3*med)
			g := heldPath(tau)
			g.Dependence = []variate.Dependence{{Members: members, Rank: rank}}
			cases[fmt.Sprintf("rank %v, median %.3g", rank, med)] = g
		}
	}
	tau, _ := dist.NewLognormalFromMedianP90(0.25, 0.75)
	rate, _ := dist.NewGammaFromMeanSD(8, 2)
	drawn := heldPath(tau)
	drawn.Sources[0].Frequency = rate
	drawn.Dependence = []variate.Dependence{{Members: members, Rank: 0.95}}
	cases["drawn rate"] = drawn
	two := heldPath(tau)
	two.Nodes[1].Next, two.Nodes[1].NextHolding = "c", tau
	two.Nodes = append(two.Nodes, variate.Node{ID: "c", Functions: []variate.Function{Fn("z", "z", C(2))}})
	two.Dependence = []variate.Dependence{{Members: []string{"a.x", "a->b", "b->c", "b.y"}, Rank: 0.95}}
	cases["two holdings"] = two
	for name, g := range cases {
		m, e := depExact(t, g)
		if math.IsNaN(e.Var) {
			t.Errorf("%s: var NaN", name)
			continue
		}
		r := runSeries(t, g, 1<<21, 7)
		n := r.Series.Periods()
		checkBlockVar(t, name+" total", e.Var, r.Series.TotalColumn(make([]float64, n)))
		for c, cat := range m.Categories() {
			checkBlockVar(t, name+" "+cat, e.VarByCategory[c], r.Series.CategoryColumn(c, make([]float64, n)))
		}
		for k, src := range m.SourceIDs() {
			checkBlockVar(t, name+" source "+src, e.VarBySource[k], r.Series.SourceColumn(k, make([]float64, n)))
		}
	}
}

// checkSD fails when an exact variance's standard deviation is beyond
// four standard errors of the simulated one.
func checkSD(t *testing.T, what string, vr float64, st variate.Stat) {
	t.Helper()
	sd := math.Sqrt(vr)
	if se := st.StdDev / math.Sqrt(2*float64(st.N)); !(math.Abs(st.StdDev-sd) <= 4*se) {
		t.Errorf("%s: exact sd %v, simulated %v ± %v", what, sd, st.StdDev, se)
	}
}

// TestDelayMixedRate: a rate drawn once a period, gamma of mean 8 and sd
// 2, moves the rounds of a period together, so a period's total takes
// the rounds of the periods before it with their own rates: the exact
// standard deviation of the total, each category and the source is
// within its error of the simulated over 2^20 periods, at lognormal
// holdings of median 2 days, 3 months and a year.
func TestDelayMixedRate(t *testing.T) {
	rate, _ := dist.NewGammaFromMeanSD(8, 2)
	for _, med := range []float64{2.0 / 365, 0.25, 1} {
		tau, _ := dist.NewLognormalFromMedianP90(med, 3*med)
		g := heldPath(tau)
		g.Sources[0].Frequency = rate
		m, e := depExact(t, g)
		_, _, s := runSummary(t, g, 1<<20, 6)
		checkSD(t, "total", e.Var, s.Total)
		for _, c := range []string{"x", "y", "lost"} {
			checkSD(t, "category "+c, e.VarByCategory[indexOf(m.Categories(), c)], s.ByCategory[c])
		}
		checkSD(t, "source", e.VarBySource[0], s.BySource["s"])
	}
}

// TestDelayMixedRateExact: a booking of 1 a fixed 0.25 or 2.5 periods
// after its round's arrival falls in the next period by K, so with a
// rate Λ drawn once a period Var = λ + Var(Λ)·((3/4)² + (1/4)²) and
// Var(Λ)·2·(1/2)²: 8 + 4·0.625 and 8 + 4·0.5.
func TestDelayMixedRateExact(t *testing.T) {
	rate, _ := dist.NewGammaFromMeanSD(8, 2)
	for c, want := range map[float64]float64{0.25: 8 + 4*0.625, 2.5: 8 + 4*0.5} {
		g := variate.Graph{ID: "fixed",
			Sources: []variate.Source{{ID: "s", Frequency: rate, To: "a"}},
			Nodes: []variate.Node{
				{ID: "a", Next: "b", NextHolding: C(c)},
				{ID: "b", Functions: []variate.Function{Fn("v", "v", C(1))}},
			}}
		if _, e := depExact(t, g); !within(e.Var, want, 1e-9) {
			t.Errorf("holding %v: var %v, want %v", c, e.Var, want)
		}
	}
}

// TestDelayFactor: a value factor drawn once a period scales only its
// period's rounds, so with a holding the rounds booking into a period
// draw it apart: for a booking of 1 times the factor, mean 1 and sd 0.5,
// a fixed 1.5 periods after arrival, Var = λ·E[g²] + λ²·Var(g)·Σ_d K² =
// 5·1.25 + 25·0.25·0.5; and at a lognormal holding the exact standard
// deviation is within its error of the simulated.
func TestDelayFactor(t *testing.T) {
	f, _ := dist.NewLognormalFromMeanSD(1, 0.5)
	graph := func(tau dist.Continuous) variate.Graph {
		return variate.Graph{ID: "factor",
			Factors: []variate.Factor{{ID: "f", Model: f}},
			Sources: []variate.Source{{ID: "s", Frequency: C(5), To: "a"}},
			Nodes: []variate.Node{
				{ID: "a", Next: "b", NextHolding: tau},
				{ID: "b", Functions: []variate.Function{{Name: "v", Category: "v", Model: C(1), Factor: "f"}}},
			}}
	}
	if _, e := depExact(t, graph(C(1.5))); !within(e.Var, 5*1.25+25*0.25*0.5, 1e-9) {
		t.Errorf("fixed holding: var %v, want %v", e.Var, 5*1.25+25*0.25*0.5)
	}
	tau, _ := dist.NewLognormalFromMedianP90(2, 6)
	g := graph(tau)
	_, e := depExact(t, g)
	_, _, s := runSummary(t, g, 1<<20, 7)
	checkSD(t, "lognormal holding", e.Var, s.Total)
}

// TestDelayPeriodDraws: a value, a probability and a holding drawn once
// a period, each on its own and together with a drawn rate: the exact
// variance of the total and each category is within four errors of the
// simulated over 2^21 periods, the error from 64 blocks of the run, as
// draws shared by a period and its neighbours leave σ/√(2N) far short.
func TestDelayPeriodDraws(t *testing.T) {
	tau, _ := dist.NewLognormalFromMedianP90(0.25, 0.75)
	short, _ := dist.NewLognormalFromMedianP90(2.0/365, 6.0/365)
	q, _ := dist.NewBeta(3, 5)
	rate, _ := dist.NewGammaFromMeanSD(8, 2)
	value := heldPath(tau)
	value.Nodes[1].Functions[0].Scope = variate.PerPeriod
	prob := heldPath(tau)
	prob.Nodes[0].Next, prob.Nodes[0].NextHolding = "", nil
	prob.Nodes[0].Outcomes = []variate.Outcome{{To: "b", P: q, Holding: tau}, {To: "c", Holding: short}}
	prob.Nodes = append(prob.Nodes, variate.Node{ID: "c", Functions: []variate.Function{Fn("z", "y", C(2))}})
	holding := heldPath(tau)
	holding.Nodes[0].NextScope = variate.PerPeriod
	all := prob
	all.Nodes = append([]variate.Node(nil), prob.Nodes...)
	all.Nodes[0].Outcomes = []variate.Outcome{{To: "b", P: q, Holding: tau, HoldingScope: variate.PerPeriod}, {To: "c", Holding: short}}
	all.Nodes[1].Functions = []variate.Function{{Name: "y", Category: "y", Model: prob.Nodes[1].Functions[0].Model, Scope: variate.PerPeriod}}
	all.Sources = []variate.Source{{ID: "s", Frequency: rate, To: "a"}}
	for name, g := range map[string]variate.Graph{"value": value, "probability": prob, "holding": holding, "all": all} {
		m, e := depExact(t, g)
		r := runSeries(t, g, 1<<21, 8)
		n := r.Series.Periods()
		checkBlockVar(t, name+" total", e.Var, r.Series.TotalColumn(make([]float64, n)))
		for c, cat := range m.Categories() {
			checkBlockVar(t, name+" category "+cat, e.VarByCategory[c], r.Series.CategoryColumn(c, make([]float64, n)))
		}
	}
}

// runSeries runs g for a fixed length with its series kept.
func runSeries(t *testing.T, g variate.Graph, periods int64, seed uint64) *variate.Result {
	t.Helper()
	cfg := variate.Config{Periods: periods, AllowUnconverged: true, Workers: 4, Series: true}
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), seed)
	if err != nil {
		t.Fatal(err)
	}
	res := *r
	return &res
}

// checkBlockVar fails when an exact variance is beyond four standard
// errors of the series' variance, the error from the spread of 64
// blocks' variances about the whole series' mean.
func checkBlockVar(t *testing.T, what string, vr float64, x []float64) {
	t.Helper()
	const blocks = 64
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))
	per := len(x) / blocks
	var s1, s2 float64
	for b := range blocks {
		bv := 0.0
		for _, v := range x[b*per : (b+1)*per] {
			bv += (v - mean) * (v - mean)
		}
		bv /= float64(per)
		s1, s2 = s1+bv, s2+bv*bv
	}
	got := s1 / blocks
	se := math.Sqrt((s2/blocks - got*got) / (blocks - 1))
	if !(math.Abs(got-vr) <= 4*se) {
		t.Errorf("%s: exact var %v, simulated %v ± %v", what, vr, got, se)
	}
}

// TestDelayNotComputed: two holdings drawn once a period on one path,
// holdings in lockstep, or at rank 0.99 over 2 days, all but points
// given the group's normal, and a schedule beside a value factor or a value
// drawn once a period leave the variance with holding times NaN.
func TestDelayNotComputed(t *testing.T) {
	tau, _ := dist.NewLognormalFromMedianP90(0.25, 0.75)
	twice := heldPath(tau)
	twice.Nodes[0].NextScope = variate.PerPeriod
	twice.Nodes[1].Next, twice.Nodes[1].NextHolding, twice.Nodes[1].NextScope = "c", tau, variate.PerPeriod
	twice.Nodes = append(twice.Nodes, variate.Node{ID: "c", Functions: []variate.Function{Fn("z", "y", C(1))}})
	lockstep := heldPath(tau)
	lockstep.Dependence = []variate.Dependence{{Members: []string{"a.x", "a->b", "b.y"}, Rank: 1}}
	short, _ := dist.NewLognormalFromMedianP90(2.0/365, 6.0/365)
	near := heldPath(short)
	near.Dependence = []variate.Dependence{{Members: []string{"a.x", "a->b", "b.y"}, Rank: 0.99}}
	f, _ := dist.NewLognormalFromMeanSD(1, 0.5)
	factor := scheduled(heldPath(tau))
	factor.Factors = []variate.Factor{{ID: "f", Model: f}}
	factor.Nodes[1].Functions[0].Factor = "f"
	period := scheduled(heldPath(tau))
	period.Nodes[1].Functions[0].Scope = variate.PerPeriod
	for name, g := range map[string]variate.Graph{"twice": twice, "lockstep": lockstep, "near lockstep, 2 days": near, "schedule and factor": factor, "schedule and period value": period} {
		if _, e := depExact(t, g); !math.IsNaN(e.Var) {
			t.Errorf("%s: var %v, want NaN", name, e.Var)
		}
	}
}

// scheduled puts g's first source on a schedule of four values.
func scheduled(g variate.Graph) variate.Graph {
	g.Schedules = append(g.Schedules, variate.Schedule{ID: "season", Values: []float64{0.2, 1, 1.8, 1}})
	g.Sources[0].Schedule = "season"
	return g
}

// schedulePath: 5 rounds a period, on a schedule, book 1 after the
// holding τ.
func schedulePath(values []float64, tau dist.Continuous) variate.Graph {
	return variate.Graph{ID: "seasonal",
		Schedules: []variate.Schedule{{ID: "season", Values: values}},
		Sources:   []variate.Source{{ID: "s", Frequency: C(5), To: "a", Schedule: "season"}},
		Nodes: []variate.Node{
			{ID: "a", Next: "b", NextHolding: tau},
			{ID: "b", Functions: []variate.Function{Fn("v", "v", C(1))}},
		}}
}

// TestDelayScheduleExact: on a schedule alternating 0.5 and 1.5, a
// booking a fixed half period after arrival takes half its period's
// rounds and half the last's, a mean of 5 at every place, so Var = 5;
// a period later it takes the last period's alone, Var = 5 + 25·0.25.
// A rate drawn independently each period from the schedule's values
// would give 5 + 25·0.25·(0.25 + 0.25) at the half period.
func TestDelayScheduleExact(t *testing.T) {
	for c, want := range map[float64]float64{0.5: 5, 1: 5 + 25*0.25} {
		if _, e := depExact(t, schedulePath([]float64{0.5, 1.5}, C(c))); !within(e.Var, want, 1e-9) {
			t.Errorf("holding %v: var %v, want %v", c, e.Var, want)
		}
	}
}

// TestDelaySchedule: schedules of 2, 4 and 12 values with lognormal
// holdings of median 2 days, 3 months and a year; a drawn rate, a
// holding drawn once a period and two sources on schedules of their own,
// by category and source: the exact variance, the cycle's, is within four
// errors of 2^21 simulated periods.
func TestDelaySchedule(t *testing.T) {
	cases := map[string]variate.Graph{}
	for _, vals := range [][]float64{{0.5, 1.5}, {0.2, 1, 1.8, 1}, {0.4, 0.5, 0.7, 1, 1.3, 1.6, 1.8, 1.6, 1.3, 1, 0.7, 0.5}} {
		for _, med := range []float64{2.0 / 365, 0.25, 1} {
			tau, _ := dist.NewLognormalFromMedianP90(med, 3*med)
			cases[fmt.Sprintf("%d values, median %.3g", len(vals), med)] = schedulePath(vals, tau)
		}
	}
	tau, _ := dist.NewLognormalFromMedianP90(0.25, 0.75)
	rate, _ := dist.NewGammaFromMeanSD(8, 2)
	mixed := scheduled(heldPath(tau))
	mixed.Sources[0].Frequency = rate
	cases["drawn rate"] = mixed
	holding := scheduled(heldPath(tau))
	holding.Nodes[0].NextScope = variate.PerPeriod
	cases["holding once a period"] = holding
	two := scheduled(heldPath(tau))
	two.Schedules = append(two.Schedules, variate.Schedule{ID: "shift", Values: []float64{1.5, 0.5, 1}})
	two.Sources = append(two.Sources, variate.Source{ID: "t", Frequency: C(3), To: "b", Schedule: "shift"})
	cases["two schedules"] = two
	for name, g := range cases {
		m, e := depExact(t, g)
		r := runSeries(t, g, 1<<21, 5)
		n := r.Series.Periods()
		checkBlockVar(t, name+" total", e.Var, r.Series.TotalColumn(make([]float64, n)))
		for c, cat := range m.Categories() {
			checkBlockVar(t, name+" "+cat, e.VarByCategory[c], r.Series.CategoryColumn(c, make([]float64, n)))
		}
		for k, src := range m.SourceIDs() {
			checkBlockVar(t, name+" source "+src, e.VarBySource[k], r.Series.SourceColumn(k, make([]float64, n)))
		}
	}
}

// TestDelayPlan: with holdings of a year, neighbouring periods move
// together, and the plan sizes the total's error by the long-run
// variance per period, λ·E[B²] of a round's whole booking B: the
// variance of sums over 1,000 periods, per period, over 2^23 periods
// is it within 6%, four of its standard errors, where the per-period
// variance is 60% below.
func TestDelayPlan(t *testing.T) {
	tau, _ := dist.NewLognormalFromMedianP90(1, 3)
	g := heldPath(tau)
	cfg := variate.Config{Periods: 1 << 23, AllowUnconverged: true, TargetRSE: 0.001, Series: true}
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 2)
	if err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	e, _ := m.Analytic()
	const block = 1000
	var s1, s2, k float64
	for start := int64(0); start+block <= r.Series.Periods(); start += block {
		sum := 0.0
		for p := start; p < start+block; p++ {
			sum += r.Series.Total(p)
		}
		s1, s2, k = s1+sum, s2+sum*sum, k+1
	}
	observed := (s2/k - s1*s1/k/k) / block
	for _, row := range s.Plan {
		if row.Criterion != "target-rse" || row.Driver != "total" {
			continue
		}
		planned := float64(row.PeriodsNeeded) * 1e-6 * e.Mean * e.Mean
		if !within(planned, observed, 0.06) {
			t.Errorf("planned long-run variance %v, blocks' %v (per-period %v)", planned, observed, e.Var)
		}
		return
	}
	t.Error("no target-rse row for the total")
}

// TestDelayBoundedHolding: a PERT holding, its density a power of the
// distance to each end of its support, which falls inside a cell, over
// a period or over days: the variance is the fixed holding's averaged
// over the law, by Simpson's rule on its density over 2^20 panels, to
// 1e-8, the grids settling to 1e-9.
func TestDelayBoundedHolding(t *testing.T) {
	for _, pts := range [][3]float64{{0.1, 0.3, 1.2}, {1.0 / 365, 3.0 / 365, 10.0 / 365}} {
		p, _ := dist.NewPERT(pts[0], pts[1], pts[2], dist.DefaultPERTShape)
		_, e := depExact(t, heldPath(p))
		const panels = 1 << 20
		lo, hi := pts[0], pts[2]
		h := (hi - lo) / panels
		f := func(x float64) float64 { return math.Exp(p.LogDensity(x)) * heldConstExact(x) }
		want := 0.0
		for i := 1; i < panels; i++ {
			want += float64(2+2*(i%2)) * f(lo+float64(i)*h)
		}
		want *= h / 3
		if !within(e.Var, want, 1e-8) {
			t.Errorf("PERT %v: var %v, want %v", pts, e.Var, want)
		}
	}
}

// TestDelayExamples: the examples with holding times and drawn rates or
// a schedule, the pump station's bounded holdings among them, have an exact
// variance of the total, each category and each source within four
// errors of 2^20 simulated periods.
func TestDelayExamples(t *testing.T) {
	for _, f := range []string{"pump_station", "delays", "sites", "line", "settlement"} {
		cm, err := config.Load("../../examples/" + f + ".yaml")
		if err != nil {
			t.Fatal(err)
		}
		m, e := depExact(t, cm.Graph)
		if math.IsNaN(e.Var) {
			t.Errorf("%s: no exact variance", f)
			continue
		}
		r := runSeries(t, cm.Graph, 1<<20, 3)
		n := r.Series.Periods()
		checkBlockVar(t, f+" total", e.Var, r.Series.TotalColumn(make([]float64, n)))
		for c, cat := range m.Categories() {
			checkBlockVar(t, f+" "+cat, e.VarByCategory[c], r.Series.CategoryColumn(c, make([]float64, n)))
		}
		for k, s := range m.SourceIDs() {
			checkBlockVar(t, f+" source "+s, e.VarBySource[k], r.Series.SourceColumn(k, make([]float64, n)))
		}
	}
}
