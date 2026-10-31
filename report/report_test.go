package report_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
	"code.eitanity.com/variate/report"
)

// TestJSONWithHistogram: the JSON with the drawing histograms and the
// warnings decodes, the linear bands and the empty periods sum to one,
// the edges are linear, and the log bands are geometric and hold no
// more than the linear ones.
func TestJSONWithHistogram(t *testing.T) {
	m, err := config.Load("../examples/compound.yaml")
	if err != nil {
		t.Fatal(err)
	}
	m.Config.Periods, m.Config.Workers = 5000, 1
	c, warnings, err := m.Graph.Compile(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	var sum variate.Summary
	w.Summarize(&sum)
	top := sum.Total.P99 * 1.05
	rep := &report.Report{C: c, R: r, Sum: &sum, Hist: report.LinearHistogram(w, top, 50, c.LatticeUnit()), LogHist: report.LogHistogram(w, top/1e4, top, 50), Warnings: warnings}
	var buf bytes.Buffer
	bw := bufio.NewWriter(&buf)
	rep.WriteJSON(bw)
	if err := bw.Flush(); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Histogram struct {
			Edges []float64 `json:"edges"`
			Mass  []float64 `json:"mass"`
		} `json:"histogram"`
		LogHistogram struct {
			Edges []float64 `json:"edges"`
			Mass  []float64 `json:"mass"`
		} `json:"log_histogram"`
		Warnings []string `json:"warnings"`
	}
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v\n%s", err, buf.String())
	}
	if len(got.Histogram.Edges) != 51 || len(got.Histogram.Mass) != 51 {
		t.Fatalf("lengths %d %d", len(got.Histogram.Edges), len(got.Histogram.Mass))
	}
	total := 0.0
	for _, x := range got.Histogram.Mass {
		total += x
	}
	zero := float64(r.Total.Zero) / float64(r.Periods)
	if zero == 0 || math.Abs(total+zero-1) > 1e-9 {
		t.Errorf("bands sum to %g with %g empty", total, zero)
	}
	logTotal := 0.0
	for _, x := range got.LogHistogram.Mass {
		logTotal += x
	}
	e := got.LogHistogram.Edges
	if len(e) < 3 || len(got.LogHistogram.Mass) != len(e) || logTotal > total+1e-9 || logTotal < 0.5*total {
		t.Errorf("log bands: %d edges, mass %g of %g", len(e), logTotal, total)
	}
	if r1, r2 := e[1]/e[0], e[2]/e[1]; math.Abs(r1-r2) > 1e-9*r1 {
		t.Errorf("log edges not geometric: %v", e[:3])
	}
	if step := got.Histogram.Edges[1]; math.Abs(got.Histogram.Edges[50]-50*step) > 1e-9*got.Histogram.Edges[50] {
		t.Errorf("edges not linear: %v", got.Histogram.Edges[:3])
	}
	if len(got.Warnings) != len(warnings) {
		t.Errorf("warnings %v", got.Warnings)
	}
}

// TestBoundsBelowSpan: below the transform's first lattice point the
// bracket's upper bound is the probability of a non-empty period, not
// one, and every point's bracket holds the simulated curve.
func TestBoundsBelowSpan(t *testing.T) {
	m, err := config.Load("../examples/compound.yaml")
	if err != nil {
		t.Fatal(err)
	}
	m.Config.Periods, m.Config.Workers, m.Config.Transform = 20000, 1, true
	c, _, err := m.Graph.Compile(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 2)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := report.New(c, w, r, true)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Exceedance []struct{ X, P, StdErr, Lower, Upper float64 } `json:"exceedance"`
		Analytic   struct {
			PZero float64 `json:"pzero"`
		} `json:"analytic"`
	}
	if err := json.Unmarshal(rep.JSON(), &got); err != nil {
		t.Fatal(err)
	}
	h := rep.Tr.H
	below := 0
	for _, p := range got.Exceedance {
		if p.Lower > p.P+5*p.StdErr+1e-9 || p.Upper < p.P-5*p.StdErr-1e-9 {
			t.Errorf("x %g: bracket [%g, %g] misses %g ± %g", p.X, p.Lower, p.Upper, p.P, p.StdErr)
		}
		if p.X > 0 && p.X < h {
			below++
			if p.Upper >= 1 || p.Upper < 1-got.Analytic.PZero-1e-6 {
				t.Errorf("x %g below the span %g: upper %g, P(T > 0) = %g", p.X, h, p.Upper, 1-got.Analytic.PZero)
			}
		}
	}
	if below == 0 {
		t.Error("no exceedance point below the span")
	}
}

// TestLatticeHistogram: a model whose functions are all constants has a
// lattice unit, its linear bands each hold the same whole number of
// multiples with edges halfway between them, and consecutive interior
// bands do not alternate in height the way bands catching two then
// three atoms would; a model with a continuous function has no unit.
func TestLatticeHistogram(t *testing.T) {
	m, err := config.Load("../examples/filter.yaml")
	if err != nil {
		t.Fatal(err)
	}
	m.Config.Periods, m.Config.Workers = 20000, 1
	c, _, err := m.Graph.Compile(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	unit := c.LatticeUnit()
	if unit <= 0 {
		t.Fatalf("lattice unit %g", unit)
	}
	w, err := c.NewWorkspace(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), 3); err != nil {
		t.Fatal(err)
	}
	var sum variate.Summary
	w.Summarize(&sum)
	h := report.LinearHistogram(w, sum.Total.P99*1.05, 50, unit)
	per := math.Round((h.Edges[2] - h.Edges[1]) / unit)
	for i := 1; i < len(h.Edges); i++ {
		k := (h.Edges[i]/unit - 0.5)
		if math.Abs(k-math.Round(k)) > 1e-9 || (i > 1 && math.Abs((h.Edges[i]-h.Edges[i-1])/unit-per) > 1e-9) {
			t.Errorf("edge %d at %g is not %g multiples past a half unit", i, h.Edges[i], per)
		}
	}
	peak, at := 0.0, 0
	for i, x := range h.Mass[:50] {
		if x > peak {
			peak, at = x, i
		}
	}
	for i := max(1, at-3); i < min(49, at+3); i++ {
		if h.Mass[i] < 0.5*h.Mass[i-1] && h.Mass[i] < 0.5*h.Mass[i+1] {
			t.Errorf("band %d dips to %g between %g and %g", i, h.Mass[i], h.Mass[i-1], h.Mass[i+1])
		}
	}
	m2, err := config.Load("../examples/compound.yaml")
	if err != nil {
		t.Fatal(err)
	}
	c2, _, err := m2.Graph.Compile(m2.Config)
	if err != nil {
		t.Fatal(err)
	}
	if u := c2.LatticeUnit(); u != 0 {
		t.Errorf("compound lattice unit %g", u)
	}
}

// TestReachOrder: the reach is written source by source in model order
// with nodes in topological order, so a chain reads top to bottom.
func TestReachOrder(t *testing.T) {
	m, err := config.Load("../examples/delays.yaml")
	if err != nil {
		t.Fatal(err)
	}
	m.Config.Periods, m.Config.Workers = 2000, 1
	c, _, err := m.Graph.Compile(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(m.Config)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := report.New(c, w, r, false)
	if err != nil {
		t.Fatal(err)
	}
	js := string(rep.JSON())
	at := strings.Index(js, `"reach"`)
	if at < 0 {
		t.Fatal("no reach in the JSON")
	}
	reach := js[at:]
	if i, k := strings.Index(reach, `"repair"`), strings.Index(reach, `"back_in_service"`); i < 0 || k < 0 || i > k {
		t.Errorf("reach lists back_in_service before repair: %.200s", reach)
	}
	var buf bytes.Buffer
	bw := bufio.NewWriter(&buf)
	rep.WriteTable(bw)
	_ = bw.Flush()
	if !strings.Contains(buf.String(), "rounds from source reaching node") {
		t.Error("table heading missing")
	}
}

// TestPercentilesOnlyWithHistogram: a quantity without a histogram has
// no percentiles in the JSON and empty percentile cells in the CSV; the
// node entries have them once the node histograms are kept.
func TestPercentilesOnlyWithHistogram(t *testing.T) {
	for _, nodeHist := range []bool{false, true} {
		m, err := config.Load("../examples/compound.yaml")
		if err != nil {
			t.Fatal(err)
		}
		m.Config.Periods, m.Config.Workers, m.Config.NodeHist = 2000, 1, nodeHist
		c, _, err := m.Graph.Compile(m.Config)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(m.Config)
		if err != nil {
			t.Fatal(err)
		}
		r, err := w.Run(context.Background(), 1)
		if err != nil {
			t.Fatal(err)
		}
		var sum variate.Summary
		w.Summarize(&sum)
		w.Close()
		rep := &report.Report{C: c, R: r, Sum: &sum}
		var js, csv bytes.Buffer
		bw := bufio.NewWriter(&js)
		rep.WriteJSON(bw)
		cw := bufio.NewWriter(&csv)
		rep.WriteCSV(cw)
		if err := errors.Join(bw.Flush(), cw.Flush()); err != nil {
			t.Fatal(err)
		}
		var got struct {
			Total      map[string]any            `json:"total"`
			EntriesPer map[string]map[string]any `json:"entries_per"`
		}
		if err := json.Unmarshal(js.Bytes(), &got); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if _, ok := got.Total["p50"]; !ok {
			t.Errorf("node_hist %v: total lacks p50", nodeHist)
		}
		p50, ok := got.EntriesPer["loss"]["p50"]
		x, _ := p50.(float64)
		_, clipped := got.EntriesPer["loss"]["clipped"]
		if ok != nodeHist || clipped != nodeHist || (nodeHist && !(x > 0)) {
			t.Errorf("node_hist %v: entries p50 %v present %v, clipped present %v", nodeHist, p50, ok, clipped)
		}
		row := ""
		for line := range strings.Lines(csv.String()) {
			if strings.HasPrefix(line, "entries,loss,") {
				row = strings.TrimSpace(line)
			}
		}
		cells := strings.Split(row, ",")
		if len(cells) != 15 || (cells[8] == "") == nodeHist || (cells[13] == "") == nodeHist {
			t.Errorf("node_hist %v: csv row %q", nodeHist, row)
		}
	}
}
