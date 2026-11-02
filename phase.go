package variate

import (
	"errors"
	"slices"
)

// Phase is the place in the cycle a graph was fixed at by Graph.Phase:
// Index from 1 to Cycle, or 0 for a graph over its whole cycle.
type Phase struct{ Index, Cycle int }

// maxCycle caps the combined cycle of a model's schedules.
const maxCycle = 1 << 30

// Cycle is the length of the model's combined cycle, the least common
// multiple of its schedules' lengths; 0 for a model without schedules.
func (g Graph) Cycle() (int, error) {
	n := 0
	for _, sc := range g.Schedules {
		k := len(sc.Values)
		if k == 0 {
			return 0, errors.New("phase: schedule " + sc.ID + " has no values")
		}
		if n == 0 {
			n = k
			continue
		}
		// The product is formed in 64 bits so that it cannot wrap on a
		// 32-bit int before the cap is checked.
		l := int64(n/gcd(n, k)) * int64(k)
		if l > maxCycle {
			return 0, errors.New("phase: the schedules' combined cycle is longer than 2^30 periods")
		}
		n = int(l)
	}
	return n, nil
}

func gcd(a, b int) int {
	// bound: Euclid's steps, logarithmic in the smaller of the two
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// Phase returns the graph at phase i of its cycle, counted from 1: each
// schedule fixed at its value for the periods at that place, period p
// being at phase p mod the cycle + 1. Every period of a run is then that
// phase, and the exact figures describe one such period rather than a
// period taken at random from the cycle.
func (g Graph) Phase(i int) (Graph, error) {
	if g.Phased.Index != 0 {
		return Graph{}, errors.New("phase: the graph is already fixed at a phase")
	}
	n, err := g.Cycle()
	if err != nil {
		return Graph{}, err
	}
	if n == 0 {
		return Graph{}, errors.New("phase: the model has no schedules, so no cycle to take a phase of")
	}
	if i < 1 || i > n {
		return Graph{}, errors.New("phase: " + itoa(i) + " is outside the cycle of " + itoa(n) + " periods; give 1 to " + itoa(n))
	}
	g.Schedules = slices.Clone(g.Schedules)
	for k := range g.Schedules {
		sc := &g.Schedules[k]
		sc.Values = []float64{sc.Values[(i-1)%len(sc.Values)]}
	}
	g.Phased = Phase{Index: i, Cycle: n}
	return g, nil
}

// Phase is the phase the model was fixed at; Index 0 for the whole cycle.
func (m *Compiled) Phase() Phase { return m.phase }
