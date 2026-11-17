package suite

import (
	"context"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
	"code.eitanity.com/variate/report"
)

// TestQuantityAccess: any category, source or node entry count can be
// read as the total is: its row, its statistics, its exceedance curve
// and its bands, which sum to its non-empty share.
func TestQuantityAccess(t *testing.T) {
	m, err := config.Load("../../examples/pump_station.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg := variate.Config{Periods: 20000, Workers: 2, NodeHist: true}
	c, _, err := m.Graph.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	_, err = w.Run(context.Background(), 4)
	if err != nil {
		t.Fatal(err)
	}
	var sum variate.Summary
	w.Summarize(&sum)
	for _, q := range []struct{ kind, name string }{{"total", ""}, {"category", "labour"}, {"source", "wear_trip"}, {"entries", "trip"}, {"contrib", "major_repair"}} {
		row, ok := w.QuantityRow(q.kind, q.name)
		if !ok || !w.HasHistogram(row) {
			t.Fatalf("%s %s: row %d ok %v histogram %v", q.kind, q.name, row, ok, w.HasHistogram(row))
		}
		st := w.QuantityStat(row)
		ec := w.ExceedanceOf(row, nil)
		if len(ec) == 0 || ec[0].P <= 0 {
			t.Errorf("%s %s: no exceedance curve", q.kind, q.name)
		}
		h := report.LinearHistogramOf(w, row, st.P99*1.05, 40, 0)
		if h == nil {
			t.Fatalf("%s %s: no histogram", q.kind, q.name)
		}
		mass := 0.0
		for _, x := range h.Mass {
			mass += x
		}
		if mass > 1+1e-9 || mass < 0.5 {
			t.Errorf("%s %s: bands sum to %g", q.kind, q.name, mass)
		}
	}
	if _, ok := w.QuantityRow("source", "nobody"); ok {
		t.Error("an unknown source has a row")
	}
	if sum.ContribPer["major_repair"].Mean <= 0 || sum.ContribPer["auto_restart"].Mean != 0 {
		t.Errorf("contrib per node: %+v", sum.ContribPer)
	}
	s := c.Structure()
	if len(s.Nodes) != len(c.NodeIDs()) || len(s.Outcomes) != len(c.OutcomeNames()) || len(s.Sources) != 2 || s.Sources[0].To != "trip" {
		t.Errorf("structure %+v", s)
	}
}
