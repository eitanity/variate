package suite

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/report"
)

// TestLayers: on a model whose total is a Poisson count, a layer's mean
// equals the exact expectation of its payout under the Poisson law,
// the transform brackets that mean, the simulation agrees, the layer
// is a quantity with a histogram and a series column, and the report
// carries it.
func TestLayers(t *testing.T) {
	g := counting(C(3), 0, nil)
	g.Layers = []variate.Layer{
		{ID: "retained", Limit: 2},
		{ID: "excess", Attachment: 2, Limit: 3, Share: 0.5},
		{ID: "open", Attachment: 4},
	}
	const h = 100000
	cfg := variate.Config{Periods: h, Workers: 4, Series: true, Transform: true}
	c, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 5)
	if err != nil {
		t.Fatal(err)
	}
	var sum variate.Summary
	w.Summarize(&sum)
	tr := c.Transformed()
	if tr == nil || !tr.Applicable {
		t.Fatal("transform should apply")
	}
	for _, y := range c.Layers() {
		want := 0.0
		p := math.Exp(-3)
		for k := 0; k < 60; k++ {
			want += p * y.Share * math.Min(math.Max(float64(k)-y.Attachment, 0), y.Limit)
			p *= 3 / float64(k+1)
		}
		st := sum.Layers[y.ID]
		se := st.StdErr
		if z := (st.Mean - want) / se; math.Abs(z) > 5 || math.IsNaN(z) {
			t.Errorf("%s: simulated mean %g ± %g vs exact %g", y.ID, st.Mean, se, want)
		}
		lo, hi := tr.Total.LayerMean(y)
		if lo > want+1e-9 || (hi < want-1e-9 && !math.IsInf(hi, 1)) {
			t.Errorf("%s: transform bounds [%g, %g] miss the exact mean %g", y.ID, lo, hi, want)
		}
		if y.ID != "open" && (math.IsInf(hi, 1) || hi-lo > 0.05*want) {
			t.Errorf("%s: bounds [%g, %g] are loose", y.ID, lo, hi)
		}
		row, ok := w.QuantityRow("layer", y.ID)
		if !ok || !w.HasHistogram(row) {
			t.Errorf("%s: no quantity row", y.ID)
		}
	}
	col := r.Series.LayerColumn(1, make([]float64, h))
	mean := 0.0
	for _, v := range col {
		mean += v
	}
	if math.Abs(mean/h-sum.Layers["excess"].Mean) > 1e-9 {
		t.Errorf("series column mean %g vs %g", mean/h, sum.Layers["excess"].Mean)
	}
	rep, err := report.New(c, w, r, true)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Layers      map[string]struct{ Mean float64 }                 `json:"layers"`
		LayerBounds map[string]struct{ MeanLower, MeanUpper float64 } `json:"layer_bounds"`
		Series      map[string][]float64                              `json:"series"`
	}
	if err := json.Unmarshal(rep.JSON(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Layers) != 3 || len(got.LayerBounds) != 3 || len(got.Series["layer:excess"]) != h {
		t.Errorf("report layers %d bounds %d series %d", len(got.Layers), len(got.LayerBounds), len(got.Series["layer:excess"]))
	}
	bad := counting(C(3), 0, nil)
	bad.Layers = []variate.Layer{{ID: "x", Share: 2}}
	if _, _, err := bad.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "share") {
		t.Errorf("share above one: %v", err)
	}
}
