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
		{"start: { from: empty, horizon: 12 }", "start: { from: given, horizon: 12 }", "needs a state"},
		{"start: { from: empty, horizon: 12 }", "start: { from: empty, horizon: 12, state: [] }", "write from: given"},
		{"start: { from: empty, horizon: 12 }", "start: { from: given, horizon: 12, state: [] }", "at least one entry"},
		{"start: { from: empty, horizon: 12 }", "start: { from: given, horizon: 12, state: [{ count: 3 }] }", "needs node"},
		{"start: { from: empty, horizon: 12 }", "start: { from: given, horizon: 12, state: [{ node: a }] }", "needs count"},
		{"start: { from: empty, horizon: 12 }", "start: { from: given, horizon: 12, state: [{ node: a, count: 3, wait: 1 }] }", "unknown key"},
		{"start: { from: empty, horizon: 12 }", "start: { from: given, horizon: 12, state: [{ node: a, count: { constant: 3, scope: period } }] }", "scope does not apply"},
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

const givenModel = `
period: { days: 1 }
start:
  from: given
  horizon: 30
  state:
    - { node: line, source: jobs, count: { poisson: 40 }, age: { empirical: [1, 2, 5, 9] }, in_service: { constant: 12, unit: hours } }
    - { node: fix, to: done, count: 3, elapsed: { uniform: [0, 2] } }
graphs:
  - id: g
    sources:
      - { id: jobs, frequency: { constant: 2 }, to: line }
    nodes:
      - id: line
        queue: { servers: 2, work: { exponential: { mean: 0.5 } }, period_end: carry }
        outcomes:
          - { to: fix }
      - id: fix
        next: done
        next_holding: { exponential: { mean: 1 } }
      - id: done
        functions:
          - { name: f, category: c, constant: 1 }
`

// TestStartGiven: from: given reads the state's entries, a queue's with
// the ages and the time served, a holding's with the time held, in the
// model's time units, and the model compiles with them.
func TestStartGiven(t *testing.T) {
	m, err := Parse([]byte(givenModel), "")
	if err != nil {
		t.Fatal(err)
	}
	st := m.Graph.Start.State
	if len(st) != 2 {
		t.Fatalf("%d state entries, want 2", len(st))
	}
	q, h := st[0], st[1]
	if q.Node != "line" || q.Source != "jobs" || q.To != "" || q.Count.Mean() != 40 || q.Age.Mean() != 4.25 || q.InService.Mean() != 0.5 || q.Elapsed != nil {
		t.Errorf("queue entry %+v", q)
	}
	if h.Node != "fix" || h.To != "done" || h.Count.Mean() != 3 || h.Elapsed.Mean() != 1 || h.Age != nil {
		t.Errorf("holding entry %+v", h)
	}
	if _, _, err := m.Graph.Compile(m.Config); err != nil {
		t.Error(err)
	}
}
