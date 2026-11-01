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

// TestSweepWriters: a bracket prints as [lo, hi], a simulated row its
// standard errors and n/a where there are too few blocks, a clamp its
// footnote; the JSON writes a missing standard error as null.
func TestSweepWriters(t *testing.T) {
	sw := &variate.Sweep{Figure: "p99", Delta: 0.1, Seed: 3, Periods: 5000, Rows: []variate.SweepRow{
		{Input: variate.Input{Name: "node a outcome b p", Kind: variate.InputProbability}, Method: variate.MethodSimulation,
			Down: variate.Range{Lo: 9, Hi: 9}, Base: variate.Range{Lo: 10, Hi: 10}, Up: variate.Range{Lo: 12, Hi: 12}, Swing: 3, DownSE: 0.25, UpSE: math.NaN(), Clamped: true},
		{Input: variate.Input{Name: "source s frequency", Kind: variate.InputFrequency}, Method: variate.MethodTransform,
			Down: variate.Range{Lo: 8, Hi: 8.5}, Base: variate.Range{Lo: 10, Hi: 10.5}, Up: variate.Range{Lo: 11, Hi: 11.5}, Swing: 3, DownSE: math.NaN(), UpSE: math.NaN()},
	}}
	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	report.WriteSweepTable(w, sw)
	_ = w.Flush()
	table := buf.String()
	for _, want := range []string{"sweep of p99, each input × (1 ± 10%); simulated rows over 5000 periods at seed 3", "[8, 8.5]", "0.25", "n/a", " *\n", "* a probability was held"} {
		if !strings.Contains(table, want) {
			t.Errorf("table lacks %q:\n%s", want, table)
		}
	}
	var got struct {
		Periods int64
		Rows    []struct {
			Method  string
			Down    struct{ Lo, Hi float64 }
			UpSE    *float64 `json:"up_se"`
			Clamped bool
		}
	}
	if err := json.Unmarshal(report.SweepJSON(sw), &got); err != nil {
		t.Fatal(err)
	}
	if got.Periods != 5000 || got.Rows[0].UpSE != nil || !got.Rows[0].Clamped || got.Rows[1].Down.Hi != 8.5 {
		t.Errorf("json: %+v", got)
	}
}
