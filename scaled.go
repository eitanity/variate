package variate

import (
	"errors"
	"math"
	"slices"

	"code.eitanity.com/variate/dist"
)

// The kinds of input a sweep can scale.
const (
	InputFactor      = "factor"
	InputSchedule    = "schedule"
	InputFrequency   = "frequency"
	InputRepair      = "repair"
	InputFunction    = "function"
	InputProbability = "probability"
	InputHolding     = "holding"
)

// Input is one distribution of a graph that a sweep can scale, named as
// the report names things: "source <id> frequency", "source <id> repair",
// "node <id> function <name>", "node <id> outcome <target> p", "node
// <id> outcome <target> holding", "node <id> next holding", "factor
// <id>", "schedule <id>". A repeated target carries its #n as in the
// outcome names. Constants are inputs too.
type Input struct {
	Name string
	Kind string
}

// inputRef locates an input: the factor, schedule, source or node i, and
// within a node the function or outcome j (−1 for Next).
type inputRef struct {
	Input
	i, j int
}

// Inputs lists the graph's inputs in a fixed order: factors, schedules,
// sources, then per node its functions and outcomes. A source standing
// on chains has no frequency input, a remainder outcome no probability,
// and a node's only outcome in absolute mode no probability either.
func (g Graph) Inputs() []Input {
	refs := g.inputRefs()
	out := make([]Input, len(refs))
	for i, r := range refs {
		out[i] = r.Input
	}
	return out
}

func (g Graph) inputRefs() []inputRef {
	var refs []inputRef
	add := func(name, kind string, i, j int) {
		refs = append(refs, inputRef{Input: Input{Name: name, Kind: kind}, i: i, j: j})
	}
	for i, f := range g.Factors {
		add("factor "+f.ID, InputFactor, i, 0)
	}
	for i, sc := range g.Schedules {
		add("schedule "+sc.ID, InputSchedule, i, 0)
	}
	for i, s := range g.Sources {
		if len(s.Chains) == 0 && s.Frequency != nil {
			add("source "+s.ID+" frequency", InputFrequency, i, 0)
		}
		if s.Repair != nil {
			add("source "+s.ID+" repair", InputRepair, i, 0)
		}
	}
	for i, n := range g.Nodes {
		refs = n.inputRefs(refs, i)
	}
	return refs
}

// inputRefs appends node i's inputs: its functions, its continuation's
// holding, and its outcomes' probabilities and holdings.
func (n Node) inputRefs(refs []inputRef, i int) []inputRef {
	add := func(name, kind string, j int) {
		refs = append(refs, inputRef{Input: Input{Name: "node " + n.ID + " " + name, Kind: kind}, i: i, j: j})
	}
	for j, f := range n.Functions {
		add("function "+f.Name, InputFunction, j)
	}
	if n.Next != "" && n.NextHolding != nil {
		add("next holding", InputHolding, -1)
	}
	dupes := map[string]int{}
	for j, o := range n.Outcomes {
		dupes[o.To]++
		target := o.To
		if dupes[o.To] > 1 {
			target += "#" + itoa(dupes[o.To])
		}
		if o.P != nil && (n.Mode == Sequential || len(n.Outcomes) > 1) {
			add("outcome "+target+" p", InputProbability, j)
		}
		if o.Holding != nil {
			add("outcome "+target+" holding", InputHolding, j)
		}
	}
	return refs
}

// Scaled returns a copy of the graph with one input multiplied by f > 0;
// the receiver is not changed. A probability stays a probability: one
// that would pass 1 is held at 1, the fixed probabilities of an absolute
// node with a remainder are held to a sum of 1, and those of a node
// without one are rescaled together so that they still sum to 1.
// clamped reports that a hold changed the scaled value.
func (g Graph) Scaled(input string, f float64) (scaled Graph, clamped bool, err error) {
	if !(f > 0) || math.IsInf(f, 1) {
		return Graph{}, false, errors.New("scale: the factor must be finite and positive")
	}
	for _, r := range g.inputRefs() {
		if r.Name == input {
			return g.scale(r, f)
		}
	}
	return Graph{}, false, errors.New("scale: no input named " + input)
}

func (g Graph) scale(r inputRef, f float64) (Graph, bool, error) {
	var err error
	switch r.Kind {
	case InputFactor:
		g.Factors = slices.Clone(g.Factors)
		g.Factors[r.i].Model, err = scaleDist(g.Factors[r.i].Model, f)
	case InputSchedule:
		g.Schedules = slices.Clone(g.Schedules)
		sc := &g.Schedules[r.i]
		sc.Values = slices.Clone(sc.Values)
		for k := range sc.Values {
			sc.Values[k] *= f
		}
	case InputFrequency:
		g.Sources = slices.Clone(g.Sources)
		g.Sources[r.i].Frequency, err = scaleDist(g.Sources[r.i].Frequency, f)
	case InputRepair:
		g.Sources = slices.Clone(g.Sources)
		g.Sources[r.i].Repair, err = scaleDist(g.Sources[r.i].Repair, f)
	default:
		return g.scaleNode(r, f)
	}
	return g, false, err
}

// scaleNode scales a function, holding or probability of node r.i.
func (g Graph) scaleNode(r inputRef, f float64) (Graph, bool, error) {
	g.Nodes = slices.Clone(g.Nodes)
	n := &g.Nodes[r.i]
	var err error
	switch {
	case r.Kind == InputFunction:
		n.Functions = slices.Clone(n.Functions)
		n.Functions[r.j].Model, err = scaleDist(n.Functions[r.j].Model, f)
	case r.Kind == InputHolding && r.j < 0:
		n.NextHolding, err = scaleCont(n.NextHolding, f)
	case r.Kind == InputHolding:
		n.Outcomes = slices.Clone(n.Outcomes)
		n.Outcomes[r.j].Holding, err = scaleCont(n.Outcomes[r.j].Holding, f)
	default:
		n.Outcomes = slices.Clone(n.Outcomes)
		clamped, perr := scaleProbability(n, r.j, f)
		return g, clamped, perr
	}
	return g, false, err
}

// scaleProbability scales outcome j's probability in place, in a node
// whose outcomes are already a copy.
func scaleProbability(n *Node, j int, f float64) (bool, error) {
	p := n.Outcomes[j].P
	if !isFixed(p) {
		d, clamped, err := scaleUncertain(p, f)
		n.Outcomes[j].P = d
		return clamped, err
	}
	v := p.Mean()
	want := f * v
	if n.Mode == Sequential {
		setFixed(n, j, math.Min(want, 1))
		return want > 1, nil
	}
	others, remainder := 0.0, false
	for k, o := range n.Outcomes {
		switch {
		case k == j:
		case o.P == nil:
			remainder = true
		default:
			others += o.P.Mean()
		}
	}
	if remainder {
		top := math.Max(1-others, 0)
		setFixed(n, j, math.Min(want, top))
		return want > top, nil
	}
	if !(others > 0) {
		// The outcome holds all the node's mass: there is nothing to
		// trade with, so it stays.
		return want != v, nil
	}
	// No remainder: the others take up the change in proportion.
	got := math.Min(want, 1)
	ratio := (1 - got) / others
	for k := range n.Outcomes {
		if k != j {
			setFixed(n, k, n.Outcomes[k].P.Mean()*ratio)
		}
	}
	setFixed(n, j, got)
	return want > 1, nil
}

// setFixed sets a fixed probability.
func setFixed(n *Node, j int, v float64) {
	c, _ := dist.NewConstant(v)
	n.Outcomes[j].P = c
}

// scaleUncertain scales a drawn probability, capping the draws at 1
// when the scaled support passes it.
func scaleUncertain(p dist.Distribution, f float64) (dist.Distribution, bool, error) {
	s, err := dist.NewScaled(p, f)
	if err != nil {
		return nil, false, err
	}
	lo, hi := s.Support()
	switch {
	case hi <= 1:
		return s, false, nil
	case lo >= 1:
		c, _ := dist.NewConstant(1)
		return c, true, nil
	}
	capped, err := dist.NewCapped(s, 1)
	if err != nil {
		return nil, false, err
	}
	return capped, true, nil
}

func scaleDist(d dist.Distribution, f float64) (dist.Distribution, error) {
	s, err := dist.NewScaled(d, f)
	if err != nil {
		return nil, err
	}
	return s, nil
}

func scaleCont(d dist.Continuous, f float64) (dist.Continuous, error) {
	s, err := dist.NewScaled(d, f)
	if err != nil {
		return nil, err
	}
	return s, nil
}
