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
	for i, x := range xs {
		check("P(T ≥ "+strconv.FormatFloat(x, 'g', -1, 64)+")", share[i].StdDev(), shareSE[i].Mean)
	}
	for i, name := range []string{"p50", "p90", "p95", "p99", "p999"} {
		check(name, quant[i].StdDev(), quantSE[i].Mean)
	}
}
