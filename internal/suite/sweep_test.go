package suite

import (
	"context"
	"encoding/json"
	"math"
	"math/rand/v2"
	"strconv"
	"testing"
	"time"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
	"code.eitanity.com/variate/report"
)

// Numerical range sweeps: extreme but valid parameters must give numbers
// (or declared infinities), consistent histograms, exact scaling, and
// JSON that decodes. Nothing here checks accuracy beyond what a short
// run can; the point is that nothing degrades silently.

// compoundWith is Poisson(rate) rounds each booking one draw of value.
func compoundWith(rate dist.Distribution, value dist.Distribution) variate.Graph {
	return variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: rate, To: "a"}},
		Nodes:   []variate.Node{{ID: "a", Functions: []variate.Function{Fn("x", "k", value)}}},
	}
}

// runSummary compiles, runs a fixed length and summarises.
func runSummary(t *testing.T, g variate.Graph, periods int64, seed uint64) (*variate.Compiled, *variate.Result, variate.Summary) {
	t.Helper()
	cfg := variate.Config{Periods: periods, AllowUnconverged: true, Workers: 4}
	c, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), seed)
	if err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	return c, r, s
}

func finiteStat(t *testing.T, what string, s variate.Stat) {
	t.Helper()
	for name, x := range map[string]float64{"mean": s.Mean, "sd": s.StdDev, "min": s.Min, "max": s.Max, "p50": s.P50, "p90": s.P90, "p99": s.P99, "p999": s.P999} {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			t.Errorf("%s: %s is %g", what, name, x)
		}
	}
}

// TestSweepRates: rates from 1e-6 to 1e6 per period.
func TestSweepRates(t *testing.T) {
	for _, rate := range []float64{1e-6, 1e-3, 0.5, 9.999, 21.9, 22, 100, 1e4, 1e6} {
		what := "rate " + strconv.FormatFloat(rate, 'g', -1, 64)
		periods := int64(2000)
		if rate >= 1e4 {
			periods = 20
		}
		g := compoundWith(C(rate), Pert(1, 2, 4))
		c, r, s := runSummary(t, g, periods, 1)
		e, _ := c.Analytic()
		if !near(e.Mean, rate*(1+8+4)/6, 1e-12) || math.IsNaN(e.Var) || e.Var < 0 || e.RoundsPerPeriod != rate {
			t.Errorf("%s: exact mean %g var %g rounds %g", what, e.Mean, e.Var, e.RoundsPerPeriod)
		}
		finiteStat(t, what, s.Total)
		var events uint64
		for _, x := range r.Events {
			events += x
		}
		if z := (float64(events) - rate*float64(periods)) / math.Sqrt(rate*float64(periods)); rate*float64(periods) >= 10 && math.Abs(z) > 4.5 {
			t.Errorf("%s: %d events in %d periods, z=%.2f", what, events, periods, z)
		}
		if rate*float64(periods) < 1e-2 && (events != 0 || s.Total.Mean != 0) {
			t.Errorf("%s: %d events, mean %g", what, events, s.Total.Mean)
		}
	}
}

// TestSweepValueScale: scaling every value by s scales every statistic
// by s, to rounding, at the same seed.
func TestSweepValueScale(t *testing.T) {
	base := func(scale float64) variate.Graph {
		ln, _ := dist.NewLognormalFromMedianP90(2*scale, 9*scale)
		g := Diamond()
		g.Nodes[1].Functions[0].Model = Pert(scale, 2*scale, 3*scale)
		g.Nodes[3].Functions[0].Model = ln
		return g
	}
	_, _, ref := runSummary(t, base(1), 5000, 2)
	for _, scale := range []float64{1e-9, 1e-3, 1e6, 1e12} {
		what := "scale " + strconv.FormatFloat(scale, 'g', -1, 64)
		c, _, s := runSummary(t, base(scale), 5000, 2)
		e, _ := c.Analytic()
		if !near(e.Mean, scale*refMean(t), 1e-12) {
			t.Errorf("%s: exact mean %g vs %g", what, e.Mean, scale*refMean(t))
		}
		finiteStat(t, what, s.Total)
		if !near(s.Total.Mean, scale*ref.Total.Mean, 1e-12) || !near(s.Total.Max, scale*ref.Total.Max, 1e-12) {
			t.Errorf("%s: mean %g max %g vs %g %g", what, s.Total.Mean, s.Total.Max, scale*ref.Total.Mean, scale*ref.Total.Max)
		}
		// Quantiles come from log histograms of fixed relative width, so
		// they scale to within that width.
		for name, pair := range map[string][2]float64{"p50": {s.Total.P50, ref.Total.P50}, "p99": {s.Total.P99, ref.Total.P99}} {
			if !near(pair[0], scale*pair[1], 3e-3) {
				t.Errorf("%s: %s %g vs %g", what, name, pair[0], scale*pair[1])
			}
		}
	}
}

func refMean(t *testing.T) float64 {
	t.Helper()
	ln, _ := dist.NewLognormalFromMedianP90(2, 9)
	g := Diamond()
	g.Nodes[1].Functions[0].Model = Pert(1, 2, 3)
	g.Nodes[3].Functions[0].Model = ln
	_, e := exactOf(t, g)
	return e.Mean
}

// TestSweepHoldingAtWindow: holding times just below, at and above a
// whole number of periods, with a per-time cost, never breach the window
// and keep the mean.
func TestSweepHoldingAtWindow(t *testing.T) {
	for _, tau := range []float64{0.999, 1, 1.001, 3.999, 4, 4.001, 50} {
		what := "holding " + strconv.FormatFloat(tau, 'g', -1, 64)
		g := variate.Graph{
			Sources: []variate.Source{{ID: "s", Frequency: C(3), To: "a"}},
			Nodes: []variate.Node{
				{ID: "a", Functions: []variate.Function{{Name: "rate", Category: "k", Model: C(10), Basis: variate.PerTime}},
					Outcomes: []variate.Outcome{{To: "b", Holding: C(tau)}}},
				{ID: "b", Functions: []variate.Function{Fn("x", "k", C(1))}},
			},
		}
		c, r, s := runSummary(t, g, 3000, 3)
		e, _ := c.Analytic()
		if !near(e.Mean, 3*(10*tau+1), 1e-12) || !near(e.Contrib[0], 3*10*tau, 1e-12) || !near(e.Contrib[1], 3, 1e-12) {
			t.Errorf("%s: exact mean %g contributions %v", what, e.Mean, e.Contrib)
		}
		if c.Spill() < int(math.Ceil(tau)) {
			t.Errorf("%s: window %d shorter than the holding time", what, c.Spill())
		}
		finiteStat(t, what, s.Total)
		if math.Abs(s.Total.Mean-e.Mean) > 0.05*e.Mean {
			t.Errorf("%s: simulated mean %g vs exact %g (beyond horizon %g)", what, s.Total.Mean, e.Mean, r.BeyondHorizon)
		}
	}
}

// TestSweepFamilyEdges: every family at its awkward parameters has
// finite or declared-infinite moments, a consistent CDF and quantile, and
// draws inside its support.
func TestSweepFamilyEdges(t *testing.T) {
	must := func(d dist.Distribution, err error) dist.Distribution {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	cases := map[string]dist.Distribution{
		"pert peak at low":      must(dist.NewPERT(1, 1, 4, dist.DefaultPERTShape)),
		"pert peak at high":     must(dist.NewPERT(1, 4, 4, dist.DefaultPERTShape)),
		"pert tiny span":        must(dist.NewPERT(1, 1+1e-9, 1+2e-9, dist.DefaultPERTShape)),
		"gamma shape 1":         must(dist.NewGamma(1, 2)),
		"gamma shape 0.999":     must(dist.NewGamma(0.999, 2)),
		"gamma shape 0.05":      must(dist.NewGamma(0.05, 2)),
		"beta 0.5 0.5":          must(dist.NewBeta(0.5, 0.5)),
		"beta 50 0.5":           must(dist.NewBeta(50, 0.5)),
		"pareto alpha 1.001":    must(dist.NewPareto(1, 1.001)),
		"pareto alpha 2":        must(dist.NewPareto(1, 2)),
		"pareto alpha 2.001":    must(dist.NewPareto(1, 2.001)),
		"gpd xi 0.49":           must(dist.NewGPD(0, 1, 0.49)),
		"gpd xi 0.5":            must(dist.NewGPD(0, 1, 0.5)),
		"gpd xi 0.99":           must(dist.NewGPD(0, 1, 0.99)),
		"gpd xi -0.5":           must(dist.NewGPD(0, 1, -0.5)),
		"lognormal sigma 3":     must(dist.NewLognormal(0, 3)),
		"lognormal sigma 1e-6":  must(dist.NewLognormal(1, 1e-6)),
		"truncated 1e-9 mass":   must(dist.NewTruncated(must(dist.NewLognormal(0, 1)), 10, 11)),
		"capped at the median":  must(dist.NewCapped(must(dist.NewLognormal(0, 1)), 1)),
		"capped far out":        must(dist.NewCapped(must(dist.NewLognormal(0, 1)), 1e6)),
		"weibull shape 0.3":     must(dist.NewWeibull(0.3, 1)),
		"exponential tiny mean": must(dist.NewExponential(1e-12)),
		"uniform wide":          must(dist.NewUniform(0, 1e15)),
	}
	r := rand.New(rand.NewPCG(1, 1))
	buf := make([]float64, 2000)
	for name, d := range cases {
		lo, hi := d.Support()
		mean, vr := d.Mean(), d.Var()
		if math.IsNaN(mean) || math.IsNaN(vr) || lo > hi || mean < lo || (mean > hi && !math.IsInf(mean, 1)) {
			t.Errorf("%s: mean %g var %g support [%g, %g]", name, mean, vr, lo, hi)
		}
		d.Fill(r, buf)
		for _, x := range buf {
			if math.IsNaN(x) || x < lo || x > hi {
				t.Errorf("%s: draw %g outside [%g, %g]", name, x, lo, hi)
				break
			}
		}
		for _, p := range []float64{1e-9, 0.01, 0.5, 0.99, 1 - 1e-9} {
			x := d.Quantile(p)
			if math.IsNaN(x) || x < lo || x > hi {
				t.Errorf("%s: quantile(%g) = %g", name, p, x)
				continue
			}
			if c := d.CDF(x); !math.IsInf(x, 0) && math.Abs(c-p) > 1e-6 && math.Abs(c-p) > 1e-6*p {
				if lo != hi && !(c >= p && d.CDF(math.Nextafter(x, lo)) <= p) { // an atom or a step is fine
					t.Errorf("%s: CDF(quantile(%g)) = %g", name, p, c)
				}
			}
		}
	}
}

// TestSweepDegenerate: models that book nothing, or almost nothing,
// summarise and report without NaN, and their JSON decodes.
func TestSweepDegenerate(t *testing.T) {
	par, _ := dist.NewPareto(1, 1.5)
	cases := map[string]variate.Graph{
		"zero rate":          compoundWith(C(0), Pert(1, 2, 4)),
		"zero values":        compoundWith(C(2), C(0)),
		"infinite variance":  compoundWith(C(2), par),
		"unreachable branch": {Sources: []variate.Source{{ID: "s", Frequency: C(1), To: "a"}}, Nodes: []variate.Node{{ID: "a", Outcomes: []variate.Outcome{{To: "b", P: C(1)}, {To: "c"}}}, {ID: "b"}, {ID: "c", Functions: []variate.Function{Fn("x", "k", C(5))}}}},
	}
	for name, g := range cases {
		cfg := variate.Config{Periods: 500, AllowUnconverged: true, Transform: true}
		c, _, err := g.Compile(cfg)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		r, err := w.Run(context.Background(), 4)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		rep, err := report.New(c, w, r, true)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		text := rep.JSON()
		var decoded map[string]any
		if err := json.Unmarshal(text, &decoded); err != nil {
			t.Errorf("%s: JSON does not decode: %v\n%s", name, err, text)
		}
		s := rep.Sum.Total
		for field, x := range map[string]float64{"mean": s.Mean, "min": s.Min, "max": s.Max, "p50": s.P50, "p99": s.P99} {
			if math.IsNaN(x) {
				t.Errorf("%s: total %s is NaN", name, field)
			}
		}
		w.Close()
	}
}

// TestSweepTransformRange: the transform over rates and value scales on
// the smallest grid returns within a time bound, with brackets that are
// numbers in [0, 1], ordered, and non-increasing in x.
func TestSweepTransformRange(t *testing.T) {
	cfg := variate.TransformConfig{Tol: 0.25, MaxGrid: 1 << 16}
	for _, rate := range []float64{1e-3, 1, 30, 1e3} {
		for _, scale := range []float64{1e-6, 1, 1e9} {
			what := "rate " + strconv.FormatFloat(rate, 'g', -1, 64) + " scale " + strconv.FormatFloat(scale, 'g', -1, 64)
			g := compoundWith(C(rate), Pert(scale, 2*scale, 4*scale))
			c, _ := exactOf(t, g)
			done := make(chan *variate.Transformed, 1)
			go func() { tr, _ := c.Transform(cfg); done <- tr }()
			var tr *variate.Transformed
			select {
			case tr = <-done:
			case <-time.After(60 * time.Second):
				t.Fatalf("%s: transform did not return in 60 s", what)
			}
			if !tr.Applicable {
				// A rate of 1000 draws per period needs more than 2^16 points
				// at a quarter tolerance; every other case must apply, the
				// rate of 1e-3 included (its tail quantile sits at zero).
				if rate < 1e3 {
					t.Errorf("%s: refused: %s", what, tr.Reason)
				}
				continue
			}
			b := tr.Total
			prevLo, prevHi := 2.0, 2.0
			for k := range b.Lower {
				lo, hi := b.Lower[k], b.Upper[k]
				if math.IsNaN(lo) || math.IsNaN(hi) || lo < 0 || hi > 1 || lo > hi+1e-12 || lo > prevLo+1e-12 || hi > prevHi+1e-12 {
					t.Errorf("%s: bracket at k=%d [%g, %g] after [%g, %g]", what, k, lo, hi, prevLo, prevHi)
					break
				}
				prevLo, prevHi = lo, hi
			}
			if math.IsNaN(tr.PZero) || tr.PZero < 0 || tr.PZero > 1 {
				t.Errorf("%s: P(T=0) %g", what, tr.PZero)
			}
		}
	}
}
