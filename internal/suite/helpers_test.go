package suite

import (
	"os"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// The fixtures are built from valid parameters, so the constructors
// cannot fail; a zero Distribution would surface as a validation error
// anyway.

// C is a constant.
func C(v float64) dist.Constant {
	d, _ := dist.NewConstant(v)
	return d
}

// Pert is a PERT with the default shape.
func Pert(l, p, h float64) dist.Distribution {
	d, _ := dist.NewPERT(l, p, h, dist.DefaultPERTShape)
	return d
}

// Beta is a beta distribution.
func Beta(a, b float64) dist.Distribution {
	d, _ := dist.NewBeta(a, b)
	return d
}

// Fn is a function booked on entry.
func Fn(name, cat string, d dist.Distribution) variate.Function {
	return variate.Function{Name: name, Category: cat, Model: d, Basis: variate.OnEntry}
}

// Diamond: s → a; a → b (0.3) or c (0.7); b → d; c → d; d books a value.
func Diamond() variate.Graph {
	return variate.Graph{
		ID:      "diamond",
		Sources: []variate.Source{{ID: "s", Frequency: Pert(1, 2, 4), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Outcomes: []variate.Outcome{{To: "b", P: C(0.3)}, {To: "c", P: C(0.7)}}},
			{ID: "b", Next: "d", Functions: []variate.Function{Fn("fb", "labour", Pert(1, 2, 3))}},
			{ID: "c", Next: "d"},
			{ID: "d", Functions: []variate.Function{Fn("fd", "parts", C(10))}},
		},
	}
}

// SkipUnderRace skips a single-threaded test that the race detector
// can learn nothing from.
func SkipUnderRace(t *testing.T) {
	t.Helper()
	if os.Getenv("VARIATE_RACE") == "1" {
		t.Skip("single-threaded; skipped under the race detector")
	}
}

// shape indexes a compiled model's structure by number: whether each
// node is a leaf, the node each outcome leaves and enters, the range of
// each node's outcomes, and the node each source enters.
type shape struct {
	leaf      []bool
	outNode   []int
	outTarget []int
	outFrom   []int // node v's outcomes are [outFrom[v], outTo[v])
	outTo     []int
	srcTarget []int
}

func shapeOf(m *variate.Compiled) shape {
	st := m.Structure()
	id := map[string]int{}
	s := shape{leaf: make([]bool, len(st.Nodes)), outFrom: make([]int, len(st.Nodes)), outTo: make([]int, len(st.Nodes))}
	for v, n := range st.Nodes {
		id[n.ID] = v
		s.leaf[v] = n.Leaf
	}
	for j, o := range st.Outcomes {
		from := id[o.From]
		if s.outTo[from] == 0 {
			s.outFrom[from] = j
		}
		s.outTo[from] = j + 1
		s.outNode = append(s.outNode, from)
		s.outTarget = append(s.outTarget, id[o.To])
	}
	for _, src := range st.Sources {
		s.srcTarget = append(s.srcTarget, id[src.To])
	}
	return s
}
