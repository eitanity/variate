package suite

import (
	"context"
	"math"
	"strconv"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// Exact identities that mutation testing showed no test was holding:
// each kills a mutant that survived the statistical checks.

// roundSummer sums delivered round values and counts rounds.
type roundSummer struct {
	rounds int
	value  float64
}

func (s *roundSummer) Observe(r *variate.Round) error {
	s.rounds++
	s.value += r.Value
	return nil
}

// TestMutantPeriodZeroCounts: period 0's rounds are counted in Events
// (a mutant counting only p > 0 survived every z-test). Three periods,
// so that at least one has a round.
func TestMutantPeriodZeroCounts(t *testing.T) {
	obs := &roundSummer{}
	cfg := variate.Config{Periods: 3, Observer: obs, Workers: 1}
	m, _, err := Diamond().Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	if obs.rounds == 0 || obs.rounds != int(r.Events[0]) { //nolint:gosec // small count
		t.Errorf("delivered %d rounds, Events %d", obs.rounds, r.Events[0])
	}
}

// TestMutantRoundValuesSumToTotals: with no delays, the values of all
// delivered rounds sum to the period totals, per-time bookings included
// (a mutant subtracting the rate booking from the round value survived).
func TestMutantRoundValuesSumToTotals(t *testing.T) {
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(3), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{{Name: "rate", Category: "k", Model: Pert(1, 2, 4), Basis: variate.PerTime}},
				Outcomes: []variate.Outcome{{To: "b", Holding: Pert(0.1, 0.2, 0.3)}}},
			{ID: "b", Functions: []variate.Function{Fn("x", "k", C(1))}},
		},
	}
	obs := &roundSummer{}
	cfg := variate.Config{Periods: 500, Observer: obs, Series: true, BlockPeriods: 50}
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
	total := 0.0
	for _, x := range r.Series.TotalColumn(make([]float64, 500)) {
		total += x
	}
	// The burn-in's rounds book into the first periods without being
	// delivered, and the last rounds book beyond the horizon, so the two
	// sums differ by those edges only: a small fraction of one period.
	e, _ := m.Analytic()
	if math.Abs(obs.value-r.BeyondHorizon-total) > 2*e.Mean {
		t.Errorf("round values %g (beyond horizon %g) vs period totals %g", obs.value, r.BeyondHorizon, total)
	}
	if z := (obs.value/500 - e.Mean) / r.Total.StdErr(false); math.Abs(z) > 5 {
		t.Errorf("mean round value per period %g vs exact %g, z=%.2f", obs.value/500, e.Mean, z)
	}
}

// TestMutantAliasTable: a selection wide enough for the alias table (more
// than eight fixed outcomes) takes each outcome at its probability.
func TestMutantAliasTable(t *testing.T) {
	const n = 12
	g := variate.Graph{Sources: []variate.Source{{ID: "s", Frequency: C(20), To: "a"}}}
	hub := variate.Node{ID: "a"}
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += float64(i + 1)
	}
	for i := 0; i < n; i++ {
		id := "o" + strconv.Itoa(i)
		hub.Outcomes = append(hub.Outcomes, variate.Outcome{To: id, P: C(float64(i+1) / sum)})
		g.Nodes = append(g.Nodes, variate.Node{ID: id})
	}
	g.Nodes = append([]variate.Node{hub}, g.Nodes...)
	m, r := run(t, g, variate.Config{Periods: 2000}, 3)
	e, _ := m.Analytic()
	for j := range r.OutcomeCount[0] {
		want := e.OutcomeRate[j] * 2000
		if z := (float64(r.OutcomeCount[0][j]) - want) / math.Sqrt(want); math.Abs(z) > 4.5 {
			t.Errorf("outcome %d: %d taken, expected %g, z=%.2f", j, r.OutcomeCount[0][j], want, z)
		}
	}
}

// TestMutantTailPeriodsWithoutTransform: without the transform no period
// counts as beyond the tail bound.
func TestMutantTailPeriodsWithoutTransform(t *testing.T) {
	_, r := run(t, Diamond(), variate.Config{Periods: 300}, 1)
	if r.TailPeriods != 0 {
		t.Errorf("tail periods %d without a bound", r.TailPeriods)
	}
}

// TestMutantShortBlocks: blocks shorter than the window route the rows
// beyond the block into the spill, so the mean is kept (a mutant that
// closed one row too many survived with the default block size).
func TestMutantShortBlocks(t *testing.T) {
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(4), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Outcomes: []variate.Outcome{{To: "b", Holding: Pert(2, 3, 4.5)}}},
			{ID: "b", Functions: []variate.Function{Fn("x", "k", C(1))}},
		},
	}
	m, r := run(t, g, variate.Config{Periods: 20000, BlockPeriods: 2, Workers: 3}, 5)
	if m.Spill() <= 2 {
		t.Fatalf("window %d not longer than the block", m.Spill())
	}
	e, _ := m.Analytic()
	if z := (r.Total.Mean() - e.Mean) / r.Total.StdErr(false); math.Abs(z) > 4.5 {
		t.Errorf("mean %g vs exact %g, z=%.2f", r.Total.Mean(), e.Mean, z)
	}
}

// TestMutantZeroPeriods: the count of empty periods equals the periods
// whose total is exactly 0 in the series.
func TestMutantZeroPeriods(t *testing.T) {
	_, r := run(t, Diamond(), variate.Config{Periods: 1000, Series: true}, 6)
	zeros := int64(0)
	for _, x := range r.Series.TotalColumn(make([]float64, 1000)) {
		if x == 0 {
			zeros++
		}
	}
	if zeros == 0 || r.Total.Zero != zeros {
		t.Errorf("Zero %d, series has %d empty periods", r.Total.Zero, zeros)
	}
}

// outcomeCounter counts the outcomes taken by delivered rounds.
type outcomeCounter struct{ taken map[int32]uint64 }

func (c *outcomeCounter) Observe(r *variate.Round) error {
	for _, s := range r.Steps {
		if s.Outcome >= 0 {
			c.taken[s.Outcome]++
		}
	}
	return nil
}

// TestMutantOutcomeCountsIncludePeriodZero: every outcome taken by a
// delivered round is in OutcomeCount, period 0 included.
func TestMutantOutcomeCountsIncludePeriodZero(t *testing.T) {
	obs := &outcomeCounter{taken: map[int32]uint64{}}
	cfg := variate.Config{Periods: 30, Observer: obs, Workers: 1}
	g := Diamond()
	g.Sources[0].Frequency = C(5) // period 0 has rounds, bar a 0.7% chance
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	for j, n := range r.OutcomeCount[0] {
		if n != obs.taken[int32(j)] { //nolint:gosec // a few outcomes
			t.Errorf("outcome %d: counted %d, delivered %d", j, n, obs.taken[int32(j)]) //nolint:gosec // a few outcomes
		}
	}
	if len(obs.taken) == 0 {
		t.Error("no outcomes delivered")
	}
}

// TestMutantPeriodScopedPairVariance: two period-scoped values on one
// node are drawn once per period and so add covariance across the
// period's rounds; the exact variance must match the simulated one (a
// mutant flipping the Θ second-moment cross term survived).
func TestMutantPeriodScopedPairVariance(t *testing.T) {
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(4), To: "a"}},
		Nodes: []variate.Node{{ID: "a", Functions: []variate.Function{
			{Name: "x", Category: "k", Model: Pert(1, 2, 6), Scope: variate.PerPeriod},
			{Name: "y", Category: "k", Model: Pert(2, 5, 9), Scope: variate.PerPeriod},
		}}},
	}
	const periods = 40000
	m, r := run(t, g, variate.Config{Periods: periods, BlockPeriods: 500}, 9)
	e, _ := m.Analytic()
	got := r.Total.StdDev() * r.Total.StdDev()
	// The variance of a sample variance over n periods is about 2σ⁴/n for
	// this short-tailed total; allow 6 of those.
	if math.Abs(got-e.Var) > 6*math.Sqrt(2*e.Var*e.Var/periods) {
		t.Errorf("simulated variance %g vs exact %g", got, e.Var)
	}
}

// TestMutantUncertainRateVariance: with an uncertain rate the variance
// has the closed form E[λ]·E[X²] + Var[λ]·E[X]², which the between-source
// term must reproduce exactly (a mutant flipping its sign survived).
func TestMutantUncertainRateVariance(t *testing.T) {
	gamma, _ := dist.NewGamma(3, 1.5)
	x := Pert(1, 2, 6)
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: gamma, To: "a"}},
		Nodes:   []variate.Node{{ID: "a", Functions: []variate.Function{Fn("x", "k", x)}}},
	}
	_, e := exactOf(t, g)
	want := gamma.Mean()*(x.Var()+x.Mean()*x.Mean()) + gamma.Var()*x.Mean()*x.Mean()
	if !near(e.Var, want, 1e-12) || !near(e.VarBySource[0], want, 1e-12) || !near(e.VarByCategory[0], want, 1e-12) {
		t.Errorf("variance %g by source %g by category %g, closed form %g", e.Var, e.VarBySource[0], e.VarByCategory[0], want)
	}
}
