package suite

import (
	"context"
	"math"
	"strconv"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
	"code.eitanity.com/variate/stats"
)

// carryModel books a cost per day of a repair that holds for about ten
// one-day periods, so neighbouring periods are strongly correlated.
const carryModel = `
period: { days: 1 }
graphs:
  - id: g
    sources:
      - { id: failure, frequency: { gamma: { shape: 2, scale: 0.25 } }, to: repair }
    nodes:
      - id: repair
        functions:
          - { name: downtime, category: production, basis: per_time, unit: days, lognormal: { median: 100, p90: 300 } }
        outcomes:
          - { to: done, holding: { lognormal: { median: 10, p90: 30 }, unit: days } }
      - id: done
`

// TestBatchErrorUnderDelays: with correlated periods, the standard error
// of the exceedance share at fixed points and of each quantile matches
// the spread of the estimates across seeds, where the binomial error,
// which takes periods as independent, is 3.5 to 5 times too small.
func TestBatchErrorUnderDelays(t *testing.T) {
	periods, seeds, tol := int64(1<<18), 64, 0.35
	if longSuite() {
		periods, seeds, tol = 1<<20, 200, 0.25
	}
	m, err := config.Parse([]byte(carryModel), ".")
	if err != nil {
		t.Fatal(err)
	}
	m.Config.Periods, m.Config.Workers = periods, 2
	c, _, err := m.Graph.Compile(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	xs := []float64{950, 2000, 2800}
	var share, shareSE [3]stats.Welford
	var mean, meanSE stats.Welford
	var quant, quantSE [5]stats.Welford
	var sum variate.Summary
	for s := range seeds {
		w, err := c.NewWorkspace(m.Config)
		if err != nil {
			t.Fatal(err)
		}
		r, err := w.Run(context.Background(), uint64(s)+1)
		if err != nil {
			t.Fatal(err)
		}
		tot := &r.Total
		mean.Add(tot.Mean())
		meanSE.Add(tot.StdErr(true))
		for i, x := range xs {
			p := tot.Hist.Exceedance(x)
			share[i].Add(p)
			shareSE[i].Add(tot.Inflation(x) * math.Sqrt(p*(1-p)/float64(tot.Hist.N)))
		}
		w.Summarize(&sum)
		q := [5]float64{sum.Total.P50, sum.Total.P90, sum.Total.P95, sum.Total.P99, sum.Total.P999}
		for i := range q {
			quant[i].Add(q[i])
			quantSE[i].Add(sum.Total.QSE[i])
		}
		w.Close()
	}
	check := func(what string, spread, reported float64) {
		ratio := reported / spread
		t.Logf("%s: across-seed sd %.4g, reported %.4g, ratio %.3f", what, spread, reported, ratio)
		if !(math.Abs(ratio-1) <= tol) {
			t.Errorf("%s: reported error %.4g is %.3f of the across-seed sd %.4g, want within %.0f%%", what, reported, ratio, spread, 100*tol)
		}
	}
	check("mean", mean.StdDev(), meanSE.Mean)
	for i, x := range xs {
		check("P(T ≥ "+strconv.FormatFloat(x, 'g', -1, 64)+")", share[i].StdDev(), shareSE[i].Mean)
	}
	for i, name := range []string{"p50", "p90", "p95", "p99", "p999"} {
		check(name, quant[i].StdDev(), quantSE[i].Mean)
	}
}

// heavyModel books a cost per day of a holding time with a log-logistic
// tail of index 1.7, whose correlation outlasts any block: its burn-in is
// 1.7·10⁶ periods.
const heavyModel = `
period: { days: 1 }
graphs:
  - id: g
    sources:
      - { id: advisory, frequency: 0.2, to: open }
    nodes:
      - id: open
        functions:
          - { name: exposure, category: risk, basis: per_time, unit: days, constant: 10 }
        outcomes:
          - { to: patched, holding: { loglogistic: { shape: 1.7, scale: 5 }, unit: days } }
      - id: patched
`

// TestMeanErrorHeavyTail: under a holding time whose correlation outlasts
// any block, the standard error of the mean, from independent timelines,
// matches the spread of the mean across seeds, where block means gave
// 0.66 of it over 24 seeds; and the seeds' mean is the exact 2·E[D],
// E[D] = 5·(π/1.7)/sin(π/1.7).
func TestMeanErrorHeavyTail(t *testing.T) {
	seeds, tol := 32, 0.25
	if longSuite() {
		seeds, tol = 128, 0.15
	}
	m, err := config.Parse([]byte(heavyModel), ".")
	if err != nil {
		t.Fatal(err)
	}
	m.Config.Periods = 1 << 20
	c, _, err := m.Graph.Compile(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	var mean stats.Welford
	var ss float64
	for s := range seeds {
		r, err := w.Run(context.Background(), uint64(s)+1)
		if err != nil {
			t.Fatal(err)
		}
		mean.Add(r.Total.Mean())
		se := r.Total.StdErr(true)
		ss += se * se
	}
	reported := math.Sqrt(ss / float64(seeds))
	ratio := reported / mean.StdDev()
	t.Logf("across-seed sd %.4g, rms reported %.4g, ratio %.3f", mean.StdDev(), reported, ratio)
	if !(math.Abs(ratio-1) <= tol) {
		t.Errorf("reported error %.4g is %.3f of the across-seed sd %.4g, want within %.0f%%", reported, ratio, mean.StdDev(), 100*tol)
	}
	b := math.Pi / 1.7
	exact := 2 * 5 * b / math.Sin(b)
	if z := (mean.Mean - exact) / (mean.StdDev() / math.Sqrt(float64(seeds))); math.Abs(z) > 4 {
		t.Errorf("mean over seeds %.4f, exact %.4f, z %.1f", mean.Mean, exact, z)
	}
}
