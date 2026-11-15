package config

import (
	"strings"
	"testing"
)

const queueModel = `
start: { from: empty, horizon: 6 }
graphs:
  - id: g
    sources:
      - { id: s, frequency: { constant: 2 }, to: q }
    nodes:
      - id: q
        queue: { servers: 3, speed: 2.5, work: { exponential: { mean: 1 } }, period_end: reset, unfinished: late }
        outcomes:
          - { to: done }
      - id: done
      - id: late
`

// TestQueue: a node's queue is read with its servers, speed, work, and
// period end with the unfinished node.
func TestQueue(t *testing.T) {
	m, err := Parse([]byte(queueModel), "")
	if err != nil {
		t.Fatal(err)
	}
	q := m.Graph.Nodes[0].Queue
	if q == nil || q.Servers != 3 || q.Speed != 2.5 || !q.Reset || q.Unfinished != "late" || q.Work.Mean() != 1 {
		t.Errorf("queue %+v", q)
	}
}

// TestQueueErrors: a queue needs its work and period end, and a mode of
// carry or reset.
func TestQueueErrors(t *testing.T) {
	cases := []struct{ from, to, want string }{
		{"work: { exponential: { mean: 1 } }, ", "", "needs work"},
		{"period_end: reset, ", "", "needs period_end"},
		{"period_end: reset", "period_end: later", "carry"},
		{"servers: 3", "servers: 0", "outside"},
	}
	for _, c := range cases {
		_, err := Parse([]byte(strings.Replace(queueModel, c.from, c.to, 1)), "")
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%q → %q: error %v, want one containing %q", c.from, c.to, err, c.want)
		}
	}
}
