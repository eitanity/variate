package config

import "testing"

// TestThresholdParse: a threshold names its function, or takes the
// node's only one, and its two nodes resolve like any target.
func TestThresholdParse(t *testing.T) {
	const text = `
graphs:
  - id: g
    sources: [ { id: s, frequency: 1, to: a } ]
    nodes:
      - { id: a, functions: [ { name: cost, category: k, lognormal: { median: 10, p90: 40 } } ], threshold: { over: 25, above: b, below: c } }
      - { id: b }
      - { id: c }
`
	m, err := Parse([]byte(text), "")
	if err != nil {
		t.Fatal(err)
	}
	th := m.Graph.Nodes[0].Threshold
	if th == nil || th.Function != "cost" || th.Over != 25 || th.Above != "b" || th.Below != "c" {
		t.Fatalf("threshold %+v", th)
	}
	if _, _, err := m.Graph.Compile(m.Config); err != nil {
		t.Fatal(err)
	}
}
