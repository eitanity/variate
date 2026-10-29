package config

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
)

const blockModel = `
period: { days: 365 }
graphs:
  - id: plant
    sources:
      - id: cooling
        block:
          unit: hours
          series:
            - { kofn: { k: 2, n: 3, mtbf: 6000, mttr: 36 } }
            - { kofn: { k: 1, n: 2, mtbf: 12000, mttr: 24 } }
            - { mtbf: 30000, mttr: 8 }
        to: outage
    nodes:
      - id: outage
        functions:
          - { name: callout, category: labour, constant: 500 }
`

// TestBlockSource: a block becomes the source's chains, in the order
// written with rates per period, recorded on the model with the
// stationary rate, system MTBF and unavailability.
func TestBlockSource(t *testing.T) {
	m, err := Parse([]byte(blockModel), "")
	if err != nil {
		t.Fatal(err)
	}
	src := m.Graph.Sources[0]
	want := []variate.Chain{
		{K: 2, N: 3, Fail: 8760.0 / 6000, Repair: 8760.0 / 36},
		{K: 1, N: 2, Fail: 8760.0 / 12000, Repair: 8760.0 / 24},
		{K: 1, N: 1, Fail: 8760.0 / 30000, Repair: 8760.0 / 8},
	}
	if src.Frequency != nil || len(src.Chains) != len(want) {
		t.Fatalf("source %+v", src)
	}
	for i, c := range src.Chains {
		if c.K != want[i].K || c.N != want[i].N || math.Abs(c.Fail-want[i].Fail) > 1e-12*c.Fail || math.Abs(c.Repair-want[i].Repair) > 1e-12*c.Repair {
			t.Errorf("chain %d %+v, want %+v", i, c, want[i])
		}
	}
	rate, u := variate.ChainFigures(want)
	b := m.Blocks[0]
	if len(m.Blocks) != 1 || b.Source != "cooling" || math.Abs(b.Rate-rate) > 1e-12*rate || math.Abs(b.Unavailability-u) > 1e-15 || math.Abs(b.MTBF-8760/rate) > 1e-9*b.MTBF {
		t.Errorf("block %+v, want rate %g unavailability %g", b, rate, u)
	}
}

// TestBlockErrors: each malformed block names its fault.
func TestBlockErrors(t *testing.T) {
	cases := []struct{ text, want string }{
		{strings.Replace(blockModel, "period: { days: 365 }\n", "", 1), "period: declared"},
		{strings.Replace(blockModel, "          unit: hours\n", "", 1), "needs unit"},
		{strings.Replace(blockModel, "        block:", "        frequency: 2\n        block:", 1), "not both"},
		{strings.Replace(blockModel, "        block:", "        every: 2\n        block:", 1), "no every"},
		{strings.Replace(blockModel, "        block:", "        repair: { constant: 1, unit: days }\n        block:", 1), "no repair"},
		{strings.Replace(blockModel, "{ mtbf: 30000, mttr: 8 }", "{ mttr: 8, kofn: { k: 1, n: 1, mtbf: 5 } }", 1), "mttr goes with mtbf"},
		{strings.Replace(blockModel, "k: 2, n: 3", "k: 4, n: 3", 1), "outside"},
		{strings.Replace(blockModel, "mtbf: 30000, mttr: 8", "mtbf: 0", 1), "must be positive"},
		{strings.Replace(blockModel, "mtbf: 30000, mttr: 8", "mtbf: 30000, mttr: -1", 1), "zero or positive"},
		{strings.Replace(blockModel, "{ mtbf: 30000, mttr: 8 }", "{ series: [] }", 1), "at least one"},
		{strings.Replace(blockModel, "{ mtbf: 30000, mttr: 8 }", "{ mtbf: 1, kofn: { k: 1, n: 1, mtbf: 1 } }", 1), "one of mtbf, kofn or series"},
	}
	for _, c := range cases {
		_, err := Parse([]byte(c.text), "")
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("want %q, got %v", c.want, err)
		}
	}
}

// TestNestedSeries: a series inside a series flattens to the chains of
// its parts in the order written, in any unit mix.
func TestNestedSeries(t *testing.T) {
	nested := strings.Replace(blockModel, `            - { mtbf: 30000, mttr: 8 }`, `            - { unit: days, series: [ { mtbf: 1250, mttr: 0.333333333333333333 }, { series: [ { kofn: { k: 1, n: 2, mtbf: 500, mttr: 1 } } ] } ] }`, 1)
	m, err := Parse([]byte(nested), "")
	if err != nil {
		t.Fatal(err)
	}
	got := m.Graph.Sources[0].Chains
	if len(got) != 4 || got[2].N != 1 || got[3].N != 2 || math.Abs(got[2].Fail-8760.0/30000) > 1e-9 || math.Abs(got[3].Fail-8760.0/12000) > 1e-9 {
		t.Errorf("chains %+v", got)
	}
}

// TestSeriesWithoutRepairIsSources: a series of components replaced at
// once is a Poisson source: the sum of the sources it stands for, with
// the exact variance.
func TestSeriesWithoutRepairIsSources(t *testing.T) {
	blocks := `
period: { hours: 1000 }
graphs:
  - id: g
    sources:
      - { id: line, block: { unit: hours, series: [ { mtbf: 400 }, { mtbf: 250 }, { kofn: { k: 2, n: 2, mtbf: 1000 } } ] }, to: down }
    nodes:
      - { id: down, functions: [ { name: cost, category: c, constant: 10 } ] }
`
	sources := `
period: { hours: 1000 }
graphs:
  - id: g
    sources:
      - { id: a, frequency: 2.5, to: down }
      - { id: b, frequency: 4, to: down }
      - { id: c1, frequency: 1, to: down }
      - { id: c2, frequency: 1, to: down }
    nodes:
      - { id: down, functions: [ { name: cost, category: c, constant: 10 } ] }
`
	var means, vars [2]float64
	for i, text := range []string{blocks, sources} {
		m, err := Parse([]byte(text), "")
		if err != nil {
			t.Fatal(err)
		}
		if i == 0 && (m.Graph.Sources[0].Chains != nil || m.Graph.Sources[0].Frequency == nil) {
			t.Fatalf("an unrepaired block should be a frequency: %+v", m.Graph.Sources[0])
		}
		c, _, err := m.Graph.Compile(variate.Config{})
		if err != nil {
			t.Fatal(err)
		}
		e, err := c.Analytic()
		if err != nil {
			t.Fatal(err)
		}
		means[i], vars[i] = e.Mean, e.Var
	}
	if math.Abs(means[0]-means[1]) > 1e-12*means[0] || math.Abs(vars[0]-vars[1]) > 1e-12*vars[0] {
		t.Errorf("block mean %g var %g, sources mean %g var %g", means[0], vars[0], means[1], vars[1])
	}
}
