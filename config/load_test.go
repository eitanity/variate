package config

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

func parse(t *testing.T, text string) *Model {
	t.Helper()
	m, err := Parse([]byte(text), "")
	if err != nil {
		t.Fatalf("parse: %v\n%s", err, text)
	}
	return m
}

func parseErr(t *testing.T, text, want string) {
	t.Helper()
	_, err := Parse([]byte(text), "")
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("want error containing %q, got %v", want, err)
	}
}

const minimal = `
graphs:
  - id: g
    sources:
      - { id: s, frequency: 2, to: a }
    nodes:
      - id: a
        functions:
          - { name: f, category: k, pert: [1, 2, 4] }
`

func TestLoadExamples(t *testing.T) {
	for _, f := range []string{"pump_station.yaml", "templates.yaml"} {
		m, err := Load(filepath.Join("testdata", f))
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		c, _, err := m.Graph.Compile(m.Config)
		if err != nil {
			t.Fatalf("%s: compile: %v", f, err)
		}
		if e, _ := c.Analytic(); !(e.Mean > 0) {
			t.Errorf("%s: mean %g", f, e.Mean)
		}
	}
	m, _ := Load("testdata/pump_station.yaml")
	if m.Config.Periods != 1000000 || m.Config.Seed != 7 || m.PeriodHours != 8760 {
		t.Errorf("settings %+v period %g", m.Config, m.PeriodHours)
	}
	// Holding time in hours divides by 8760: the lognormal median 4 h.
	var fc variate.Node
	for _, n := range m.Graph.Nodes {
		if n.ID == "fault_check" {
			fc = n
		}
	}
	if fc.Mode != variate.Sequential || len(fc.Outcomes) != 3 {
		t.Fatalf("fault_check %+v", fc)
	}
	if med := fc.Outcomes[0].Holding.Quantile(0.5); math.Abs(med-4.0/8760) > 1e-12 {
		t.Errorf("minor repair median holding %g periods, want %g", med, 4.0/8760)
	}
}

func TestLoadRules(t *testing.T) {
	parseErr(t, strings.Replace(minimal, "category: k", "category: k, colour: red", 1), `unknown key "colour"`)
	parseErr(t, "graphs: []\nbogus: 1\n", `unknown key "bogus"`)
	parseErr(t, strings.Replace(minimal, "    nodes:", "    id: g2\n    nodes:", 1), "duplicate key")
	parseErr(t, "graphs: []\n", "no graphs")
	parseErr(t, "", "empty")
	parseErr(t, strings.Replace(minimal, "pert: [1, 2, 4]", "pert: [1, 2, 4], uniform: [1, 2]", 1), "name a family")
	parseErr(t, strings.Replace(minimal, "pert: [1, 2, 4]", "pert: [1, 2]", 1), "expected 3 numbers")
	parseErr(t, strings.Replace(minimal, "pert: [1, 2, 4]", "pert: [1, 2, x]", 1), `"x" is not a number`)
	parseErr(t, strings.Replace(minimal, "frequency: 2", "frequency: { beta: [2, 3], scope: period }", 1), "scope is not allowed on a frequency")
	parseErr(t, strings.Replace(minimal, "pert: [1, 2, 4]", "pert: [1, 2, 4], unit: days", 1), "unit is not allowed on a value")
	parseErr(t, strings.Replace(minimal, "pert: [1, 2, 4]", "uniform: [1, 2], t: 3", 1), "t applies to pert")
	big := make([]byte, maxFileSize+1)
	if _, err := Parse(big, ""); err == nil || !strings.Contains(err.Error(), "16 MB") {
		t.Errorf("size cap: %v", err)
	}
	// Error text names the key and line, never the value's surroundings.
	_, err := Parse([]byte(minimal+"          - { name: g, category: k, secret: 1 }\n"), "")
	if err == nil || !strings.Contains(err.Error(), "line 10") || strings.Contains(err.Error(), "pert") {
		t.Errorf("error %v", err)
	}
}

func TestProbabilityScope(t *testing.T) {
	g := `
graphs:
  - id: g
    sources:
      - { id: s, frequency: 2, to: a }
    nodes:
      - id: a
        outcomes:
          - { to: b, p: { beta: [2, 5], scope: event } }
          - { to: c }
      - id: b
      - id: c
`
	parseErr(t, g, "always drawn per period")
	m := parse(t, strings.Replace(g, ", scope: event", "", 1))
	if _, ok := m.Graph.Nodes[0].Outcomes[0].P.(dist.Beta); !ok {
		t.Errorf("p %T", m.Graph.Nodes[0].Outcomes[0].P)
	}
	m = parse(t, strings.Replace(g, "{ beta: [2, 5], scope: event }", "0.25", 1))
	if m.Graph.Nodes[0].Outcomes[0].P.Mean() != 0.25 {
		t.Errorf("fixed p %g", m.Graph.Nodes[0].Outcomes[0].P.Mean())
	}
}

// TestUnits: every continuous family scales exactly by the unit factor;
// count families refuse a unit; a unit without period: is refused.
func TestUnits(t *testing.T) {
	families := map[string]string{
		"constant":    "constant: 3",
		"uniform":     "uniform: [1, 3]",
		"loguniform":  "loguniform: [1, 3]",
		"triangular":  "triangular: [1, 2, 3]",
		"pert":        "pert: [1, 2, 3]",
		"logpert":     "logpert: [1, 2, 3]",
		"truncnormal": "truncnormal: { mean: 2, sd: 1, lo: 0, hi: 5 }",
		"lognormal":   "lognormal: { median: 2, p90: 5 }",
		"lognormal2":  "lognormal: { mu: 0.5, sigma: 0.4 }",
		"lognormal3":  "lognormal: { mean: 2, sd: 1 }",
		"exponential": "exponential: { mean: 2 }",
		"exprate":     "exponential: { rate: 0.5 }",
		"gamma":       "gamma: { shape: 2, scale: 1.5 }",
		"gammarate":   "gamma: { shape: 2, rate: 1.5 }",
		"gammamean":   "gamma: { mean: 2, sd: 1 }",
		"weibull":     "weibull: { shape: 1.5, scale: 2 }",
		"pareto":      "pareto: { min: 1, alpha: 3 }",
		"gpd":         "gpd: { loc: 1, scale: 2, shape: 0.2 }",
		"categorical": "categorical: { values: [1, 2, 5], p: [0.2, 0.3, 0.5] }",
		"empirical":   "empirical: [1, 2, 2, 5]",
		"truncated":   "lognormal: { median: 2, p90: 5 }, truncate: [1, 4]",
		"capped":      "pert: [1, 2, 3], cap: 2.5",
	}
	const factor = 8760.0
	for name, fam := range families {
		plain := holdingModel("", fam, "")
		scaled := holdingModel("period: { days: 365 }", fam, ", unit: hours")
		a := parse(t, plain).Graph.Nodes[0].Outcomes[0].Holding
		b := parse(t, scaled).Graph.Nodes[0].Outcomes[0].Holding
		for _, p := range []float64{0.01, 0.1, 0.5, 0.9, 0.99} {
			qa, qb := a.Quantile(p)/factor, b.Quantile(p)
			// Log families scale through a log parameter, which costs a
			// few ulps; the others are exact to one.
			if math.Abs(qa-qb) > 1e-14*qa {
				t.Errorf("%s q%g: %.17g scaled %.17g", name, p, qa, qb)
			}
		}
		if ma, mb := a.Mean()/factor, b.Mean(); math.Abs(ma-mb) > 1e-12*ma {
			t.Errorf("%s mean %g scaled %g", name, ma, mb)
		}
	}
	for _, fam := range []string{"poisson: 3", "negbin: { mean: 2, size: 1 }", "geometric: 0.3", "bernoulli: 0.5", "binomial: { n: 3, p: 0.5 }", "beta: [2, 3]"} {
		parseErr(t, holdingModel("period: { days: 365 }", fam, ", unit: hours"), "unit is not allowed on")
	}
	parseErr(t, holdingModel("", "pert: [1, 2, 3]", ", unit: days"), "unit: days used but period: not declared")
	parse(t, holdingModel("", "pert: [1, 2, 3]", ", unit: periods"))
	parseErr(t, "period: { days: 1, hours: 2 }\n"+minimal, "exactly one")
	parseErr(t, "period: { years: 1 }\n"+minimal, `unknown key "years"`)
}

func holdingModel(period, family, unit string) string {
	return period + `
graphs:
  - id: g
    sources:
      - { id: s, frequency: 2, to: a }
    nodes:
      - id: a
        outcomes:
          - { to: b, holding: { ` + family + unit + ` } }
      - id: b
`
}

// TestPerTimeUnits: the same per_time model under two period lengths has
// the same analytic mean and a different window.
func TestPerTimeUnits(t *testing.T) {
	model := func(period string) string {
		return period + `
graphs:
  - id: g
    sources:
      - { id: s, frequency: 2, to: a }
    nodes:
      - id: a
        functions:
          - { name: down, category: k, basis: per_time, unit: days, constant: 100 }
        outcomes:
          - { to: b, holding: { pert: [1, 3, 10], unit: days } }
      - id: b
`
	}
	year := parse(t, model("period: { days: 365 }"))
	week := parse(t, model("period: { weeks: 1 }"))
	cy, _, _ := year.Graph.Compile(variate.Config{})
	cw, _, _ := week.Graph.Compile(variate.Config{})
	ey, _ := cy.Analytic()
	ew, _ := cw.Analytic()
	if math.Abs(ey.Mean-ew.Mean) > 1e-9*ey.Mean || math.Abs(ey.Mean-2*100*(1+3*4+10)/6.0) > 1e-9*ey.Mean {
		t.Errorf("means %g %g", ey.Mean, ew.Mean)
	}
	if cw.Spill() <= cy.Spill() {
		t.Errorf("windows year %d week %d", cy.Spill(), cw.Spill())
	}
}

func TestSettings(t *testing.T) {
	text := `
periods: 5000
seed: 3
workers: 2
min_hits: 10
tail: 0.99
pace: 1.5s
allow_unconverged: true
transform: true
transform_grid: 131072
transform_tol: 0.01
transform_work: 1000000
transform_bytes: 1073741824
` + minimal
	m := parse(t, text)
	c := m.Config
	if c.Periods != 5000 || c.Seed != 3 || c.Workers != 2 || c.MinHits != 10 || c.TailP != 0.99 || c.Pace.Seconds() != 1.5 || !c.AllowUnconverged {
		t.Errorf("%+v", c)
	}
	if !c.Transform || c.TransformGrid != 131072 || c.TransformTol != 0.01 || c.TransformWork != 1000000 || c.TransformBytes != 1<<30 {
		t.Errorf("transform settings %+v", c)
	}
	parseErr(t, "transform_grid: -1\n"+minimal, "outside")
	parseErr(t, "periods: -1\n"+minimal, "outside")
	parseErr(t, "pace: soon\n"+minimal, "not a duration")
	parseErr(t, "series: maybe\n"+minimal, "not true or false")
	var c2 variate.Config
	if err := SetSetting(&c2, "target-rse", "0.02"); err != nil || c2.TargetRSE != 0.02 {
		t.Errorf("flag form: %v %+v", err, c2)
	}
	if err := SetSetting(&c2, "colour", "red"); err == nil {
		t.Error("unknown setting accepted")
	}
}

func TestEmpiricalFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "data.csv"), []byte("1, 2\n3\n4.5\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "out.csv")
	if err := os.WriteFile(outside, []byte("1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "link.csv")); err != nil {
		t.Skip("no symlinks")
	}
	model := func(file string) string {
		return `
graphs:
  - id: g
    sources:
      - { id: s, frequency: 2, to: a }
    nodes:
      - id: a
        functions:
          - { name: f, category: k, empirical: { file: "` + file + `" } }
`
	}
	m, err := Parse([]byte(model("data.csv")), dir)
	if err != nil {
		t.Fatal(err)
	}
	if mean := m.Graph.Nodes[0].Functions[0].Model.Mean(); math.Abs(mean-10.5/4) > 1e-12 {
		t.Errorf("mean %g", mean)
	}
	for file, want := range map[string]string{
		"/etc/passwd":  "relative",
		"../data.csv":  "..",
		"link.csv":     "outside",
		"missing.csv":  "does not exist",
		"sub/../x.csv": "..",
	} {
		_, err := Parse([]byte(model(file)), dir)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: want %q, got %v", file, want, err)
		}
	}
	if _, err := Parse([]byte(model("data.csv")), ""); err == nil {
		t.Error("data file accepted without a directory")
	}
	huge := filepath.Join(dir, "huge.csv")
	f, err := os.Create(huge) //nolint:gosec // test temp dir
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(maxDataSize + 1); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := Parse([]byte(model("huge.csv")), dir); err == nil || !strings.Contains(err.Error(), "64 MB") {
		t.Errorf("size cap: %v", err)
	}
}

func TestMergeGraphs(t *testing.T) {
	text := `
graphs:
  - id: one
    sources: [ { id: s, frequency: 1, to: a } ]
    nodes: [ { id: a, functions: [ { name: f, category: k, constant: 1 } ] } ]
  - id: two
    sources: [ { id: s, frequency: 2, to: a } ]
    nodes: [ { id: a, functions: [ { name: f, category: k, constant: 10 } ] } ]
`
	m := parse(t, text)
	if len(m.Graph.Sources) != 2 || m.Graph.Sources[1].ID != "two.s" || m.Graph.Sources[1].To != "two.a" || m.Graph.Nodes[0].ID != "one.a" {
		t.Errorf("%+v", m.Graph)
	}
	c, _, err := m.Graph.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if e, _ := c.Analytic(); math.Abs(e.Mean-21) > 1e-12 {
		t.Errorf("portfolio mean %g", e.Mean)
	}
}

func TestTemplates(t *testing.T) {
	m, err := Load("testdata/templates.yaml")
	if err != nil {
		t.Fatal(err)
	}
	ids := map[string]variate.Node{}
	for _, n := range m.Graph.Nodes {
		ids[n.ID] = n
	}
	a, b := ids["pump_a_repair.assess"], ids["pump_b_repair.assess"]
	if a.Outcomes[0].To != "new_pump" || a.Outcomes[1].To != "recommission" || a.Outcomes[0].P.Mean() != 0.2 {
		t.Errorf("instance a %+v", a)
	}
	if b.Outcomes[0].P.Mean() != 0.35 {
		t.Errorf("override not applied: %+v", b)
	}
	if m.Graph.Sources[0].To != "pump_a_repair.assess" {
		t.Errorf("source target %s", m.Graph.Sources[0].To)
	}
	base := `
templates:
  - id: inner
    entry: x
    exits: [done]
    nodes:
      - { id: x, functions: [ { name: f, category: k, constant: 1 } ], next: done }
  - id: outer
    entry: start
    exits: [out]
    nodes:
      - { id: start, next: i.x }
    instances:
      - { template: inner, as: i, wire: { done: out } }
graphs:
  - id: g
    sources: [ { id: s, frequency: 1, to: o.start } ]
    instances:
      - { template: outer, as: o, wire: { out: end }, override: { "i.x.functions[0]": { constant: 5 } } }
    nodes: [ { id: end } ]
`
	m = parse(t, base)
	ids = map[string]variate.Node{}
	for _, n := range m.Graph.Nodes {
		ids[n.ID] = n
	}
	x := ids["o.i.x"]
	if x.Next != "end" || x.Functions[0].Model.Mean() != 5 || ids["o.start"].Next != "o.i.x" {
		t.Errorf("nested: %+v %+v", x, ids["o.start"])
	}
	parseErr(t, strings.Replace(base, "wire: { out: end }", "wire: {}", 1), `exit "out" of template "outer" is not wired`)
	parseErr(t, strings.Replace(base, "wire: { out: end }", "wire: { out: end, extra: end }", 1), `no exit "extra"`)
	parseErr(t, strings.Replace(base, "template: outer, as: o", "template: nope, as: o", 1), `unknown template "nope"`)
	parseErr(t, strings.Replace(base, `"i.x.functions[0]"`, `"i.y.functions[0]"`, 1), "names no node")
	parseErr(t, strings.Replace(base, `"i.x.functions[0]"`, `"i.x.functions[3]"`, 1), "index out of range")
	// Depth: a template that instantiates itself.
	loop := `
templates:
  - id: self
    entry: a
    nodes: [ { id: a, next: s.a } ]
    instances: [ { template: self, as: s } ]
graphs:
  - id: g
    sources: [ { id: s, frequency: 1, to: r.a } ]
    instances: [ { template: self, as: r } ]
`
	parseErr(t, loop, "nested deeper than 8")
	// Count: 1025 instances.
	var sb strings.Builder
	sb.WriteString("templates:\n  - id: leaf\n    entry: a\n    nodes: [ { id: a } ]\ngraphs:\n  - id: g\n    sources: [ { id: s, frequency: 1, to: i0.a } ]\n    instances:\n")
	for i := 0; i <= maxInstances; i++ {
		sb.WriteString("      - { template: leaf, as: i" + strings.Repeat("0", 0) + itoa(i) + " }\n")
	}
	parseErr(t, sb.String(), "more than 1024 instances")
}

func itoa(i int) string { return strconv.Itoa(i) }

func FuzzLoad(f *testing.F) {
	for _, file := range []string{"testdata/pump_station.yaml", "testdata/templates.yaml"} {
		data, err := os.ReadFile(file) //nolint:gosec // test data
		if err != nil {
			f.Fatal(err)
		}
		f.Add(data)
	}
	f.Add([]byte(minimal))
	f.Add([]byte("graphs: [ { id: g, sources: [ { id: s, frequency: { pert: [1, 2, 3], unit: days }, to: a } ], nodes: [ { id: a } ] } ]"))
	f.Add([]byte("x: &a [1, 2]\ngraphs: *a\n"))
	f.Add([]byte("graphs: !!binary aGk=\n"))
	f.Add([]byte("graphs: []\n---\ngraphs: []\n"))
	f.Add([]byte("graphs: " + strings.Repeat("[", 40) + strings.Repeat("]", 40) + "\n"))
	f.Add([]byte("%YAML 1.2\n---\ngraphs: []\n"))
	f.Fuzz(func(t *testing.T, data []byte) {
		m, err := Parse(data, "")
		if err != nil {
			return
		}
		_, _, _ = m.Graph.Compile(m.Config)
	})
}

// TestRenewalSourceKeys: every and repair reach the source, repair in
// periods from its unit.
func TestRenewalSourceKeys(t *testing.T) {
	text := `
period: { days: 365 }
graphs:
  - id: g
    sources:
      - { id: pallets, frequency: 20000, every: 500, to: change }
      - { id: press, frequency: 12, repair: { pert: [1, 3, 10], unit: days }, to: fix }
    nodes:
      - { id: change, functions: [ { name: filter, category: parts, constant: 80 } ] }
      - { id: fix, functions: [ { name: repair, category: labour, constant: 400 } ] }
`
	m := parse(t, text)
	if m.Graph.Sources[0].Every != 500 || m.Graph.Sources[1].Repair == nil {
		t.Fatalf("sources %+v", m.Graph.Sources)
	}
	if mean := m.Graph.Sources[1].Repair.Mean(); math.Abs(mean-(1+4*3+10)/6.0/365) > 1e-12 {
		t.Errorf("repair mean %g periods", mean)
	}
	parseErr(t, strings.Replace(text, "every: 500", "every: 0", 1), "outside")
}

// TestOutageHoldingKey: holding: outage and next_holding: outage mark
// the outcome, and a distribution still parses beside them elsewhere.
func TestOutageHoldingKey(t *testing.T) {
	text := `
period: { days: 365 }
graphs:
  - id: g
    sources:
      - { id: plant, block: { unit: hours, kofn: { k: 1, n: 2, mtbf: 1000, mttr: 10 } }, to: outage }
    nodes:
      - { id: outage, outcomes: [ { to: restored, holding: outage } ] }
      - { id: restored, next: logged, next_holding: outage }
      - { id: logged }
`
	m := parse(t, text)
	if !m.Graph.Nodes[0].Outcomes[0].OutageHolding || m.Graph.Nodes[0].Outcomes[0].Holding != nil || !m.Graph.Nodes[1].NextOutage {
		t.Errorf("nodes %+v", m.Graph.Nodes)
	}
}

// TestFactorKeys: factors parse at the top level and a source names one.
func TestFactorKeys(t *testing.T) {
	text := `
period: { days: 365 }
factors:
  - { id: climate, pert: [0.5, 1, 2] }
graphs:
  - id: g
    sources:
      - { id: a, frequency: 2, factor: climate, to: n }
      - { id: b, frequency: 3, to: n }
    nodes:
      - { id: n, functions: [ { name: one, category: c, constant: 1 } ] }
`
	m := parse(t, text)
	if len(m.Graph.Factors) != 1 || m.Graph.Factors[0].ID != "climate" || m.Graph.Sources[0].Factor != "climate" || m.Graph.Sources[1].Factor != "" {
		t.Errorf("factors %+v sources %+v", m.Graph.Factors, m.Graph.Sources)
	}
	parseErr(t, strings.Replace(text, "pert: [0.5, 1, 2]", "pert: [0.5, 1, 2], unit: days", 1), "unit is not allowed on a factor")
}

// TestScheduleKeys: schedules parse at the top level and a source names one.
func TestScheduleKeys(t *testing.T) {
	text := `
period: { days: 30 }
schedules:
  - { id: season, values: [0.5, 1, 1.5, 1] }
graphs:
  - id: g
    sources:
      - { id: a, frequency: 2, schedule: season, to: n }
    nodes:
      - { id: n, functions: [ { name: one, category: c, constant: 1 } ] }
`
	m := parse(t, text)
	if len(m.Graph.Schedules) != 1 || m.Graph.Schedules[0].ID != "season" || len(m.Graph.Schedules[0].Values) != 4 || m.Graph.Sources[0].Schedule != "season" {
		t.Errorf("schedules %+v sources %+v", m.Graph.Schedules, m.Graph.Sources)
	}
	parseErr(t, strings.Replace(text, ", values: [0.5, 1, 1.5, 1]", "", 1), "schedule season has no values")
	parseErr(t, strings.Replace(text, "1.5", "x", 1), "is not a number")
}

// TestLayerKeys: layers parse at the top level with their defaults.
func TestLayerKeys(t *testing.T) {
	text := `
period: { days: 365 }
layers:
  - { id: retained, limit: 250000 }
  - { id: reinsurance, attachment: 250000, limit: 2000000, share: 0.9 }
graphs:
  - id: g
    sources:
      - { id: a, frequency: 2, to: n }
    nodes:
      - { id: n, functions: [ { name: one, category: c, constant: 1 } ] }
`
	m := parse(t, text)
	if len(m.Graph.Layers) != 2 || m.Graph.Layers[1].Share != 0.9 || m.Graph.Layers[0].Attachment != 0 || m.Graph.Layers[0].Share != 0 {
		t.Errorf("layers %+v", m.Graph.Layers)
	}
	parseErr(t, strings.Replace(text, "share: 0.9", "share: 0.9, basis: x", 1), "unknown key")
}
