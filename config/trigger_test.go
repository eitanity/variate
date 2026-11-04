package config

import "testing"

// TestTriggerParse: an input names a node or node.function, split once
// every node is known, so IDs carrying a graph's prefix keep their
// dots; entries name nodes.
func TestTriggerParse(t *testing.T) {
	const text = `
graphs:
  - id: g
    sources: [ { id: s, frequency: 1, to: a } ]
    nodes:
      - { id: a, functions: [ { name: cost, category: k, constant: 5 } ], next: b }
      - { id: b, functions: [ { name: fee, category: k, constant: 1 }, { name: tax, category: k, constant: 2 } ] }
      - { id: t, trigger: { over: 10, inputs: [a, b.tax] } }
      - { id: u, trigger: { over: 2, entries: [a] } }
  - id: h
    sources: [ { id: s, frequency: 1, to: a } ]
    nodes:
      - { id: a, functions: [ { name: cost, category: k, constant: 5 } ] }
      - { id: t, trigger: { over: 10, inputs: [a.cost] } }
`
	m, err := Parse([]byte(text), "")
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]int{}
	for i, n := range m.Graph.Nodes {
		byID[n.ID] = i
	}
	tr := m.Graph.Nodes[byID["g.t"]].Trigger
	if tr == nil || tr.Over != 10 || len(tr.Inputs) != 2 || tr.Inputs[0].Node != "g.a" || tr.Inputs[0].Function != "" ||
		tr.Inputs[1].Node != "g.b" || tr.Inputs[1].Function != "tax" {
		t.Fatalf("trigger %+v", tr)
	}
	if u := m.Graph.Nodes[byID["g.u"]].Trigger; u == nil || len(u.Entries) != 1 || u.Entries[0] != "g.a" {
		t.Fatalf("entries trigger %+v", u)
	}
	if h := m.Graph.Nodes[byID["h.t"]].Trigger; h.Inputs[0].Node != "h.a" || h.Inputs[0].Function != "cost" {
		t.Fatalf("prefixed input %+v", h.Inputs[0])
	}
	if _, _, err := m.Graph.Compile(m.Config); err != nil {
		t.Fatal(err)
	}
}
