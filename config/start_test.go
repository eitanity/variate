package config

import (
	"strings"
	"testing"
)

const startModel = `
start: { from: empty, horizon: 12 }
repeats: 500
graphs:
  - id: g
    sources:
      - { id: s, frequency: { constant: 2 }, to: a }
    nodes:
      - id: a
        functions:
          - { name: f, category: c, constant: 1 }
`

// TestStart: start sets the graph's horizon and repeats the run's count.
func TestStart(t *testing.T) {
	m, err := Parse([]byte(startModel), "")
	if err != nil {
		t.Fatal(err)
	}
	if m.Graph.Start.Horizon != 12 || m.Config.Repeats != 500 {
		t.Errorf("horizon %d, repeats %d; want 12 and 500", m.Graph.Start.Horizon, m.Config.Repeats)
	}
}

// TestStartErrors: a start needs from: empty and a horizon, and refuses
// the settings a started run has no use for, saying why.
func TestStartErrors(t *testing.T) {
	cases := []struct{ from, to, want string }{
		{"start: { from: empty, horizon: 12 }", "start: { horizon: 12 }", "from: empty"},
		{"start: { from: empty, horizon: 12 }", "start: { from: given, horizon: 12 }", "empty"},
		{"start: { from: empty, horizon: 12 }", "start: { from: empty }", "needs a horizon"},
		{"start: { from: empty, horizon: 12 }", "start: { from: empty, horizon: 0 }", "outside"},
		{"repeats: 500", "periods: 500", "counts repeats"},
		{"repeats: 500", "burn_in: 10", "no burn-in"},
		{"repeats: 500", "block_periods: 10", "one repeat"},
		{"repeats: 500", "phase: 2", "first value of the cycle"},
	}
	for _, c := range cases {
		text := strings.Replace(startModel, c.from, c.to, 1)
		_, err := Parse([]byte(text), "")
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one containing %q", c.to, err, c.want)
		}
	}
}
