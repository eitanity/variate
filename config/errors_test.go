package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Every error the loader can return, reached from a model file. The
// expected text is the part of the message that names the fault.

func funcModel(family string) string {
	return "graphs:\n  - id: g\n    sources: [ { id: s, frequency: 1, to: a } ]\n    nodes:\n      - id: a\n        functions:\n          - { name: f, category: k, " + family + " }\n"
}

func TestLoaderErrorPaths(t *testing.T) {
	cases := []struct{ name, text, want string }{
		// walk.go
		{"map expected", "graphs: 3\n", "expected a list"},
		{"list expected", "graphs:\n  - id: g\n    nodes: 7\n", "expected a list"},
		{"map expected for node", "graphs:\n  - id: g\n    nodes: [ 3 ]\n", "expected a map"},
		{"value expected", "periods: [1]\n" + minimal, "expected a value"},
		{"empty value", "graphs:\n  - id: \"\"\n", "empty value"},
		{"not a number", "tail: abc\n" + minimal, "is not a number"},
		{"not a whole number", "periods: 1.5\n" + minimal, "not a whole number"},
		{"not a bool", "series: 2\n" + minimal, "not true or false"},
		{"not one of", "graphs:\n  - id: g\n    nodes: [ { id: a, mode: random } ]\n", "not one of"},
		{"key not scalar", "graphs:\n  - id: g\n    nodes:\n      - ? [a]\n        : 1\n", "is not a scalar"},
		{"not a map for mapping", "graphs:\n  - id: g\n    nodes: [ { id: a, functions: [ { name: f, category: k, lognormal: 3 } ] } ]\n", "expected a map"},
		{"wrong count", funcModel("uniform: [1]"), "expected 2 numbers"},
		{"list of non-numbers", funcModel("pert: [a, b, c]"), "is not a number"},
		// load.go
		{"period not positive", "period: { days: 0 }\n" + minimal, "must be positive"},
		{"period not a number", "period: { days: x }\n" + minimal, "is not a number"},
		{"periods out of range", "periods: -5\n" + minimal, "outside"},
		{"max_memory bad", "max_memory: x\n" + minimal, "not a whole number"},
		{"workers bad", "workers: -1\n" + minimal, "outside"},
		{"null document", "---\n", "expected a map"},
		{"two documents", "graphs: []\n---\ngraphs: []\n", "more than one document"},
		{"anchor", "graphs: &g []\n", "anchors are not allowed"},
		{"alias", "x: &a 1\ngraphs: *a\n", "aliases are not allowed"},
		{"custom tag", "graphs: !custom []\n", `tag "!custom" is not allowed`},
		{"binary tag", "graphs: !!binary aGk=\n", `tag "!!binary" is not allowed`},
		{"timestamp tag", "periods: !!timestamp 2001-12-14\n" + minimal, `tag "!!timestamp" is not allowed`},
		{"deep nesting", "graphs: " + strings.Repeat("[", 40) + strings.Repeat("]", 40) + "\n", "nesting deeper than 32"},
		// dist.go
		{"no family", funcModel("unit: days"), "no distribution family"},
		{"two families", funcModel("pert: [1, 2, 3], uniform: [1, 2]"), "name a family"},
		{"bad unit", funcModel("pert: [1, 2, 3], unit: fortnights"), "not one of"},
		{"bad t", funcModel("pert: [1, 2, 3], t: x"), "is not a number"},
		{"bad scope", funcModel("pert: [1, 2, 3], scope: sometimes"), "not one of"},
		{"unknown modifier", funcModel("pert: [1, 2, 3], colour: red"), `unknown key "colour"`},
		{"constant not a number", funcModel("constant: x"), "is not a number"},
		{"pert invalid", funcModel("pert: [3, 2, 1]"), "pert"},
		{"logpert invalid", funcModel("logpert: [0, 1, 2]"), "logpert"},
		{"uniform invalid", funcModel("uniform: [2, 1]"), "uniform"},
		{"loguniform invalid", funcModel("loguniform: [0, 1]"), "loguniform"},
		{"triangular invalid", funcModel("triangular: [1, 0, 2]"), "triangular"},
		{"normal invalid", funcModel("normal: { mean: 0, sd: -1 }"), "normal"},
		{"normal missing", funcModel("normal: { mean: 0 }"), `missing "sd"`},
		{"normal extra", funcModel("normal: { mean: 0, sd: 1, mode: 1 }"), `unknown key "mode"`},
		{"truncnormal invalid", funcModel("truncnormal: { mean: 0, sd: 1, lo: 5, hi: 4 }"), "truncnormal"},
		{"lognormal invalid", funcModel("lognormal: { mu: 0, sigma: -1 }"), "lognormal"},
		{"lognormal median p90 invalid", funcModel("lognormal: { median: 5, p90: 4 }"), "lognormal"},
		{"lognormal mean sd invalid", funcModel("lognormal: { mean: 0, sd: 1 }"), "lognormal"},
		{"lognormal keys", funcModel("lognormal: { mu: 0, p90: 4 }"), "lognormal takes"},
		{"exponential invalid", funcModel("exponential: { mean: 0 }"), "exponential"},
		{"exponential rate invalid", funcModel("exponential: { rate: 0 }"), "exponential"},
		{"exponential keys", funcModel("exponential: { scale: 1 }"), "exponential takes"},
		{"gamma invalid", funcModel("gamma: { shape: 0, scale: 1 }"), "gamma"},
		{"gamma rate invalid", funcModel("gamma: { shape: 1, rate: 0 }"), "gamma"},
		{"gamma mean sd invalid", funcModel("gamma: { mean: 0, sd: 1 }"), "gamma"},
		{"gamma keys", funcModel("gamma: { shape: 1 }"), "gamma takes"},
		{"beta invalid", funcModel("beta: [0, 1]"), "beta"},
		{"beta mean sd invalid", funcModel("beta: { mean: 0.5, sd: 0.6 }"), "beta"},
		{"beta keys", funcModel("beta: { alpha: 1 }"), "beta takes"},
		{"weibull invalid", funcModel("weibull: { shape: 0, scale: 1 }"), "weibull"},
		{"pareto invalid", funcModel("pareto: { min: 1, alpha: 0 }"), "pareto"},
		{"gpd invalid", funcModel("gpd: { loc: 0, scale: 0, shape: 0.1 }"), "gpd"},
		{"bernoulli invalid", funcModel("bernoulli: 2"), "bernoulli"},
		{"binomial invalid", funcModel("binomial: { n: -1, p: 0.5 }"), "binomial"},
		{"poisson invalid", funcModel("poisson: -1"), "poisson"},
		{"negbin invalid", funcModel("negbin: { r: 0, p: 0.5 }"), "negbin"},
		{"negbin mean size invalid", funcModel("negbin: { mean: 1, size: 0 }"), "negbin"},
		{"negbin keys", funcModel("negbin: { r: 1 }"), "negbin takes"},
		{"geometric invalid", funcModel("geometric: 0"), "geometric"},
		{"categorical missing values", funcModel("categorical: { p: [1] }"), "missing values"},
		{"categorical missing p", funcModel("categorical: { values: [1] }"), "missing p"},
		{"categorical bad values", funcModel("categorical: { values: 3, p: [1] }"), "expected a list"},
		{"categorical length", funcModel("categorical: { values: [1, 2], p: [1] }"), "expected 2 numbers"},
		{"categorical invalid", funcModel("categorical: { values: [1, 2], p: [0.5, 0.6] }"), "categorical"},
		{"truncate bad", funcModel("pert: [1, 2, 3], truncate: [1]"), "expected 2 numbers"},
		{"truncate invalid", funcModel("pert: [1, 2, 3], truncate: [5, 6]"), "truncate"},
		{"cap bad", funcModel("pert: [1, 2, 3], cap: x"), "is not a number"},
		{"cap invalid", funcModel("pert: [1, 2, 3], cap: 0"), "cap"},
		{"unit on value", funcModel("pert: [1, 2, 3], unit: days"), "unit is not allowed on a value"},
		{"t on uniform", funcModel("uniform: [1, 2], t: 4"), "t applies"},
		// empirical.go
		{"empirical invalid", funcModel("empirical: []"), "empirical"},
		{"empirical bad list", funcModel("empirical: [a]"), "is not a number"},
		{"empirical map keys", funcModel("empirical: { path: x }"), `unknown key "path"`},
		{"empirical map no file", funcModel("empirical: {}"), "takes a list or"},
		{"empirical file no dir", funcModel("empirical: { file: data.csv }"), "not available here"},
		// graph.go
		{"template not a list", "templates: 3\ngraphs: []\n", "expected a list"},
		{"template bad key", "templates: [ { id: t, entry: a, nodes: [], colour: 1 } ]\ngraphs: []\n", `unknown key "colour"`},
		{"template no id", "templates: [ { entry: a, nodes: [] } ]\ngraphs: []\n", "missing id"},
		{"template no entry", "templates: [ { id: t, nodes: [] } ]\ngraphs: []\n", "missing entry"},
		{"template bad exits", "templates: [ { id: t, entry: a, exits: [1, []], nodes: [] } ]\ngraphs: []\n", "expected a value"},
		{"template no nodes", "templates: [ { id: t, entry: a } ]\ngraphs: []\n", "has no nodes"},
		{"template duplicate", "templates: [ { id: t, entry: a, nodes: [] }, { id: t, entry: a, nodes: [] } ]\ngraphs: []\n", "duplicate template"},
		{"graph no id", "graphs: [ { nodes: [] } ]\n", "missing id"},
		{"graph bad description", "graphs: [ { id: g, description: [1] } ]\n", "expected a value"},
		{"source bad key", "graphs: [ { id: g, sources: [ { id: s, colour: 1 } ] } ]\n", `unknown key "colour"`},
		{"source no id", "graphs: [ { id: g, sources: [ { frequency: 1, to: a } ] } ]\n", "missing id"},
		{"source no to", "graphs: [ { id: g, sources: [ { id: s, frequency: 1 } ] } ]\n", "missing to"},
		{"source no frequency", "graphs: [ { id: g, sources: [ { id: s, to: a } ] } ]\n", "missing frequency"},
		{"source bad frequency", "graphs: [ { id: g, sources: [ { id: s, to: a, frequency: { pert: [3, 2, 1] } } ] } ]\n", "pert"},
		{"node no id", "graphs: [ { id: g, nodes: [ { next: b } ] } ]\n", "missing id"},
		{"node bad functions", "graphs: [ { id: g, nodes: [ { id: a, functions: 3 } ] } ]\n", "expected a list"},
		{"node bad outcomes", "graphs: [ { id: g, nodes: [ { id: a, outcomes: 3 } ] } ]\n", "expected a list"},
		{"node bad next", "graphs: [ { id: g, nodes: [ { id: a, next: [b] } ] } ]\n", "expected a value"},
		{"next_holding without next", "graphs: [ { id: g, nodes: [ { id: a, next_holding: 1 } ] } ]\n", "next_holding without next"},
		{"next_holding bad", "graphs: [ { id: g, nodes: [ { id: a, next: b, next_holding: { pert: [3, 2, 1] } } ] } ]\n", "pert"},
		{"function not a map", "graphs: [ { id: g, nodes: [ { id: a, functions: [ 3 ] } ] } ]\n", "expected a map"},
		{"function no name", strings.Replace(funcModel("pert: [1, 2, 3]"), "name: f, ", "", 1), "needs name and category"},
		{"function bad basis", funcModel("pert: [1, 2, 3], basis: sometimes"), "not one of"},
		{"outcome bad key", "graphs: [ { id: g, nodes: [ { id: a, outcomes: [ { to: b, colour: 1 } ] } ] } ]\n", `unknown key "colour"`},
		{"outcome no to", "graphs: [ { id: g, nodes: [ { id: a, outcomes: [ { p: 1 } ] } ] } ]\n", "missing to"},
		{"outcome bad p", "graphs: [ { id: g, nodes: [ { id: a, outcomes: [ { to: b, p: { beta: [0, 1] } } ] } ] } ]\n", "beta"},
		{"outcome bad holding", "graphs: [ { id: g, nodes: [ { id: a, outcomes: [ { to: b, holding: { pert: [3, 2, 1] } } ] } ] } ]\n", "pert"},
		{"instance bad key", "graphs: [ { id: g, instances: [ { template: t, as: i, colour: 1 } ] } ]\n", `unknown key "colour"`},
		{"instance no template", "graphs: [ { id: g, instances: [ { as: i } ] } ]\n", "missing template"},
		{"instance no as", "graphs: [ { id: g, instances: [ { template: t } ] } ]\n", "missing as"},
		{"instance bad wire", "graphs: [ { id: g, instances: [ { template: t, as: i, wire: 3 } ] } ]\n", "expected a map"},
		{"instance wire bad value", "graphs: [ { id: g, instances: [ { template: t, as: i, wire: { x: [1] } } ] } ]\n", "expected a value"},
		{"instance bad override", "graphs: [ { id: g, instances: [ { template: t, as: i, override: 3 } ] } ]\n", "expected a map"},
		{"override too short", "templates: [ { id: t, entry: a, nodes: [ { id: a } ] } ]\ngraphs: [ { id: g, instances: [ { template: t, as: i, override: { a: 1 } } ] } ]\n", "too short"},
		{"override starts with index", "templates: [ { id: t, entry: a, nodes: [ { id: a } ] } ]\ngraphs: [ { id: g, instances: [ { template: t, as: i, override: { \"a[0].p\": 1 } } ] } ]\n", "must start with a node name"},
		{"override bad path", "templates: [ { id: t, entry: a, nodes: [ { id: a } ] } ]\ngraphs: [ { id: g, instances: [ { template: t, as: i, override: { \"a.functions[x]\": 1 } } ] } ]\n", "not an override path"},
		{"override no outcomes", "templates: [ { id: t, entry: a, nodes: [ { id: a } ] } ]\ngraphs: [ { id: g, instances: [ { template: t, as: i, override: { \"a.outcomes[0].p\": 1 } } ] } ]\n", "node has no outcomes"},
		{"override no holding", "templates: [ { id: t, entry: a, nodes: [ { id: a, outcomes: [ { to: b } ] } ] } ]\ngraphs: [ { id: g, instances: [ { template: t, as: i, override: { \"a.outcomes[0].holding\": 1 } } ] } ]\n", "outcome has no holding"},
		{"override nested none", "templates: [ { id: t, entry: a, nodes: [ { id: a } ] } ]\ngraphs: [ { id: g, instances: [ { template: t, as: i, override: { \"x.a.outcomes[0].p\": 1 } } ] } ]\n", "has none"},
	}
	for _, c := range cases {
		_, err := Parse([]byte(c.text), "")
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: want %q, got %v", c.name, c.want, err)
		}
	}
}

// TestLoadFileErrors: the file-level errors of Load.
func TestLoadFileErrors(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "missing.yaml")); err == nil || !strings.Contains(err.Error(), "open model") {
		t.Errorf("missing file: %v", err)
	}
	dir := t.TempDir()
	if _, err := Load(dir); err == nil || !strings.Contains(err.Error(), "read model") {
		t.Errorf("directory as model: %v", err)
	}
	bad := filepath.Join(dir, "bad.yaml")
	if err := os.WriteFile(bad, []byte("graphs: [ { id: g, nodes: [ { id: a }\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(bad); err == nil || !strings.Contains(err.Error(), "parse model") {
		t.Errorf("malformed: %v", err)
	}
	data := filepath.Join(dir, "d.csv")
	if err := os.WriteFile(data, []byte("1,2,x\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "m.yaml")
	if err := os.WriteFile(model, []byte(funcModel("empirical: { file: d.csv }")), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(model); err == nil || !strings.Contains(err.Error(), "is not a number") {
		t.Errorf("bad data file: %v", err)
	}
	if err := os.WriteFile(data, []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(model); err == nil || !strings.Contains(err.Error(), "no numbers") {
		t.Errorf("empty data file: %v", err)
	}
	if err := os.Mkdir(filepath.Join(dir, "sub.csv"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(model, []byte(funcModel("empirical: { file: sub.csv }")), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(model); err == nil || !strings.Contains(err.Error(), "data file") {
		t.Errorf("directory as data file: %v", err)
	}
}
