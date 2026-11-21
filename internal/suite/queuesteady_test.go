package suite

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// TestQueueSteadyClosedForms: a carried queue in a steady-state run
// starts each timeline exactly in its long-run state, so its mean wait
// over 100,000 periods is Pollaczek–Khinchine's for one server of
// constant, exponential and lognormal work and Erlang C's for three
// exponential servers, within four standard errors, and its wait
// quantiles lie within four of their errors of the long-run intervals;
// each start looks back some periods.
func TestQueueSteadyClosedForms(t *testing.T) {
	expo, _ := dist.NewExponential(1.233)
	logn, _ := dist.NewLognormalFromMeanSD(1.233, 1.233)
	lam := 6.12
	pk := func(es, es2 float64) float64 { return lam * es2 / (2 * (1 - lam*es)) }
	sp := 10.85
	cases := []struct {
		name string
		g    variate.Graph
		want float64
	}{
		{"M/D/1", laneGraph(lam, 1, sp, C(1.233), 0), pk(1.233/sp, (1.233/sp)*(1.233/sp))},
		{"M/M/1", laneGraph(lam, 1, sp, expo, 0), pk(1.233/sp, 2*(1.233/sp)*(1.233/sp))},
		{"M/LN/1", laneGraph(lam, 1, sp, logn, 0), pk(1.233/sp, 2*(1.233/sp)*(1.233/sp))},
		{"M/M/3", laneGraph(lam, 3, sp/3, expo, 0), erlangCWait(lam, sp/3/1.233, 3)},
	}
	for _, c := range cases {
		q := queueRun(t, c.g, variate.Config{Periods: 100_000, Seed: 4})
		if math.Abs(q.Wait.Mean-c.want) > 4*q.Wait.StdErr {
			t.Errorf("%s: mean wait %v ± %v, closed form %v", c.name, q.Wait.Mean, q.Wait.StdErr, c.want)
		}
		if math.Abs(q.Utilisation-lam*1.233/sp) > 0.01 {
			t.Errorf("%s: utilisation %v, want ρ = %v", c.name, q.Utilisation, lam*1.233/sp)
		}
		lr := q.LongRun
		sim := [5]float64{q.Wait.P50, q.Wait.P90, q.Wait.P95, q.Wait.P99, q.Wait.P999}
		for i := range 4 {
			if gap := math.Max(lr.WaitLo[i]-sim[i], sim[i]-lr.WaitHi[i]); gap > 4*q.Wait.QSE[i] {
				t.Errorf("%s: wait quantile %d simulated %v ± %v, long-run [%v, %v]", c.name, i, sim[i], q.Wait.QSE[i], lr.WaitLo[i], lr.WaitHi[i])
			}
		}
		if !(q.LookBack >= 0) {
			t.Errorf("%s: look-back %v", c.name, q.LookBack)
		}
		t.Logf("%s: wait %.5f ± %.5f, closed form %.5f, look-back %.2f periods", c.name, q.Wait.Mean, q.Wait.StdErr, c.want, q.LookBack)
	}
}

// TestQueueSteadyRefusals: a carried queue in a steady-state run needs
// Poisson rounds at a constant rate, a load below 1 and a work with a
// finite fourth moment.
func TestQueueSteadyRefusals(t *testing.T) {
	expo, _ := dist.NewExponential(1)
	mixed := laneGraph(1, 1, 2, expo, 0)
	mixed.Sources[0].Frequency = Pert(0.5, 1, 1.5)
	pareto, _ := dist.NewPareto(0.3, 3.5)
	cases := []struct {
		name string
		g    variate.Graph
		want string
	}{
		{"drawn rate", mixed, "Poisson stream at a constant rate"},
		{"overload", laneGraph(3, 1, 2, expo, 0), "not below 1"},
		{"heavy tail", laneGraph(1, 1, 2, pareto, 0), "fourth moment"},
	}
	for _, c := range cases {
		_, _, err := c.g.Compile(variate.Config{})
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one containing %q", c.name, err, c.want)
		}
	}
}

// backlogGraph is the backlog demo's queue: 1.2 orders a period for two
// crews, each order's work lognormal of median 0.5 and p90 3, load 0.8,
// booking 1 a period over its stay.
func backlogGraph(horizon int) variate.Graph {
	logn, _ := dist.NewLognormalFromMedianP90(0.5, 3)
	return laneGraph(1.2, 2, 1, logn, horizon)
}

// TestQueueSteadyBacklog: two crews of lognormal work have no closed
// form; the steady-state run's mean wait and stay over 400,000 periods
// agree with 32 started runs of 50,000 periods each, within four of
// their joint errors. The work's tail is heavy: at seed 7 one order of
// 818 crew-months holds a crew, and the mean wait is 15.3 ± 7.5.
func TestQueueSteadyBacklog(t *testing.T) {
	steady := queueRun(t, backlogGraph(0), variate.Config{Periods: 400_000, Seed: 1})
	started := queueRun(t, backlogGraph(50_000), variate.Config{Repeats: 32, Seed: 8})
	for _, c := range []struct {
		name string
		a, b variate.ItemStat
	}{{"wait", steady.Wait, started.Wait}, {"stay", steady.Stay, started.Stay}} {
		if se := math.Hypot(c.a.StdErr, c.b.StdErr); math.Abs(c.a.Mean-c.b.Mean) > 4*se {
			t.Errorf("%s: steady %v ± %v, started %v ± %v", c.name, c.a.Mean, c.a.StdErr, c.b.Mean, c.b.StdErr)
		}
		t.Logf("%s: steady %.4f ± %.4f, started %.4f ± %.4f, look-back %.1f", c.name, c.a.Mean, c.a.StdErr, c.b.Mean, c.b.StdErr, steady.LookBack)
	}
}

// TestQueueSteadyLittle: three exponential servers' node books 1 a
// period over each stay, so its total a period is Little's law, the rate
// times Erlang C's mean stay, as the stays cross the blocks' edges,
// within four errors; the run is the same at 1 and 8 workers.
func TestQueueSteadyLittle(t *testing.T) {
	expo, _ := dist.NewExponential(1.233)
	lam, sp := 6.12, 10.85/3
	g := laneGraph(lam, 3, sp, expo, 0)
	want := lam * (erlangCWait(lam, sp/1.233, 3) + 1.233/sp)
	var totals [2]variate.Stat
	for i, workers := range []int{1, 8} {
		cfg := variate.Config{Periods: 200_000, Seed: 9, Workers: workers}
		c, _, err := g.Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Run(t.Context(), cfg.Seed); err != nil {
			t.Fatal(err)
		}
		var s variate.Summary
		w.Summarize(&s)
		w.Close()
		totals[i] = s.Total
	}
	if s := totals[0]; math.Abs(s.Mean-want) > 4*s.StdErr {
		t.Errorf("total a period %v ± %v, Little %v", s.Mean, s.StdErr, want)
	}
	if totals[0].Mean != totals[1].Mean || totals[0].StdErr != totals[1].StdErr {
		t.Errorf("1 worker %v ± %v, 8 workers %v ± %v", totals[0].Mean, totals[0].StdErr, totals[1].Mean, totals[1].StdErr)
	}
	t.Logf("total %.5f ± %.5f, Little %.5f", totals[0].Mean, totals[0].StdErr, want)
}

// TestQueueSteadyErrors: the backlog's mean wait over 100,000 periods,
// at 100 seeds, spreads as its standard errors say: their spread over
// their mean error is within [0.75, 1.33].
func TestQueueSteadyErrors(t *testing.T) {
	if testing.Short() {
		t.Skip("100 runs")
	}
	g := backlogGraph(0)
	means := make([]float64, 0, 100)
	se := 0.0
	for seed := range 100 {
		q := queueRun(t, g, variate.Config{Periods: 100_000, Seed: uint64(seed) + 100})
		means = append(means, q.Wait.Mean)
		se += q.Wait.StdErr / 100
	}
	m := 0.0
	for _, x := range means {
		m += x / 100
	}
	v := 0.0
	for _, x := range means {
		v += (x - m) * (x - m) / 99
	}
	r := math.Sqrt(v) / se
	if r < 0.75 || r > 1.33 {
		t.Errorf("spread %v over mean error %v: ratio %v", math.Sqrt(v), se, r)
	}
	t.Logf("mean %.4f, spread %.4f, mean error %.4f, ratio %.3f", m, math.Sqrt(v), se, r)
}

// TestQueueSteadyBlocks: one timeline's periods are the same whatever
// its blocks: the backlog's line, servers and the rates over its stays
// carry across each block's edge, so 20,000 periods in blocks of 64 and
// of 4,096 give the same total every period.
func TestQueueSteadyBlocks(t *testing.T) {
	var runs [2][]float64
	for i, block := range []int{64, 4096} {
		cfg := variate.Config{Periods: 20_000, Seed: 3, Series: true, BlockPeriods: block}
		c, _, err := backlogGraph(0).Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		r, err := w.Run(t.Context(), cfg.Seed)
		if err != nil {
			t.Fatal(err)
		}
		runs[i] = r.Series.TotalColumn(nil)
		w.Close()
	}
	for p := range runs[0] {
		if runs[0][p] != runs[1][p] {
			t.Fatalf("period %d: total %v in blocks of 64, %v in blocks of 4096", p, runs[0][p], runs[1][p])
		}
	}
}
