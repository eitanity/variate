package report_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/report"
)

// TestSweepWriters: a probability's range prints as probabilities and
// any other as factors, a bracket as [lo, hi], a simulated row its
// standard errors and n/a where there are too few blocks, a clamp its
// footnote; the JSON writes a missing standard error as null.
func TestSweepWriters(t *testing.T) {
	sw := &variate.Sweep{Figure: "p99", Seed: 3, Periods: 5000, Rows: []variate.SweepRow{
		{Input: variate.Input{Name: "node a outcome b p", Kind: variate.InputProbability, Range: &variate.InputRange{Low: 0.2, High: 0.4}}, Method: variate.MethodSimulation,
			Low: 2.0 / 3, High: 4.0 / 3, AtLow: variate.Range{Lo: 9, Hi: 9}, Base: variate.Range{Lo: 10, Hi: 10}, AtHigh: variate.Range{Lo: 12, Hi: 12}, Swing: 3, LowSE: 0.25, HighSE: math.NaN(), Clamped: true},
		{Input: variate.Input{Name: "source s frequency", Kind: variate.InputFrequency, Range: &variate.InputRange{Low: 1, High: 4}}, Method: variate.MethodTransform,
			Low: 0.5, High: 2, AtLow: variate.Range{Lo: 8, Hi: 8.5}, Base: variate.Range{Lo: 10, Hi: 10.5}, AtHigh: variate.Range{Lo: 11, Hi: 11.5}, Swing: 3, LowSE: math.NaN(), HighSE: math.NaN()},
	}}
	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	report.WriteSweepTable(w, sw)
	_ = w.Flush()
	table := buf.String()
	for _, want := range []string{"sweep of p99, each input with a range to its two ends; simulated rows over 5000 periods at seed 3", "p 0.2 – 0.4", "×0.5 – ×2", "[8, 8.5]", "0.25", "n/a", " *\n", "* a probability was held"} {
		if !strings.Contains(table, want) {
			t.Errorf("table lacks %q:\n%s", want, table)
		}
	}
	var got struct {
		Periods int64
		Rows    []struct {
			Method  string
			AtLow   struct{ Lo, Hi float64 } `json:"at_low"`
			HighSE  *float64                 `json:"high_se"`
			Range   struct{ Low, High float64 }
			Clamped bool
		}
	}
	if err := json.Unmarshal(report.SweepJSON(sw), &got); err != nil {
		t.Fatal(err)
	}
	if got.Periods != 5000 || got.Rows[0].HighSE != nil || !got.Rows[0].Clamped || got.Rows[1].AtLow.Hi != 8.5 || got.Rows[0].Range.High != 0.4 {
		t.Errorf("json: %+v", got)
	}
}
