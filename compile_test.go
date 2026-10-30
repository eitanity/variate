package variate

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate/dist"
)

// The fixtures are built from valid parameters, so the constructors cannot
// fail; a zero Distribution would surface as a validation error anyway.
func c(v float64) dist.Constant {
	d, _ := dist.NewConstant(v)
	return d
}

func pert(l, p, h float64) dist.Distribution {
	d, _ := dist.NewPERT(l, p, h, dist.DefaultPERTShape)
	return d
}

func beta(a, b float64) dist.Distribution {
	d, _ := dist.NewBeta(a, b)
	return d
}

func fn(name, cat string, d dist.Distribution) Function {
	return Function{Name: name, Category: cat, Model: d, Basis: OnEntry}
}

// diamond: s → a; a → b (0.3) or c (0.7); b → d; c → d; d books a value.
func diamond() Graph {
	return Graph{
		ID:      "diamond",
		Sources: []Source{{ID: "s", Frequency: pert(1, 2, 4), To: "a"}},
		Nodes: []Node{
			{ID: "a", Outcomes: []Outcome{{To: "b", P: c(0.3)}, {To: "c", P: c(0.7)}}},
			{ID: "b", Next: "d", Functions: []Function{fn("fb", "labour", pert(1, 2, 3))}},
			{ID: "c", Next: "d"},
			{ID: "d", Functions: []Function{fn("fd", "parts", c(10))}},
		},
	}
}

func mustCompile(t *testing.T, g Graph, cfg Config) (*Compiled, []Warning) {
	t.Helper()
	m, ws, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return m, ws
}

func TestCompileDiamond(t *testing.T) {
	m, ws := mustCompile(t, diamond(), Config{})
	if got := strings.Join(m.NodeIDs(), ","); got != "a,b,c,d" {
		t.Errorf("nodes %s", got)
	}
	// Fixed outcomes are ordered by descending probability with cumulative thresholds.
	if got := strings.Join(m.OutcomeNames(), ","); got != "a→c,a→b,b→d,c→d" {
		t.Errorf("outcomes %s", got)
	}
	if m.outs[0].cum != 0.7 || m.outs[1].cum != 1 || m.outs[2].cum != 1 {
		t.Errorf("cumulative thresholds %v", []float64{m.outs[0].cum, m.outs[1].cum, m.outs[2].cum})
	}
	if got := strings.Join(m.Categories(), ","); got != "labour,parts" {
		t.Errorf("categories %s", got)
	}
	if m.Spill() != 0 || m.HasDelays() {
		t.Error("no delays expected")
	}
	// Reach: d is reached with certainty, b with 0.3.
	if r := m.reach[0]; r[0] != 1 || math.Abs(r[1]-0.3) > 1e-15 || math.Abs(r[2]-0.7) > 1e-15 || math.Abs(r[3]-1) > 1e-15 {
		t.Errorf("reach %v", r)
	}
	// Pools: frequency, fb, fd; each period-scoped pool has a slot.
	if len(m.Pools()) != 3 || m.periodSlots != 1 || m.poolPeriod[0] != 0 || m.poolPeriod[1] != noIndex {
		t.Errorf("pools %d slots %d", len(m.Pools()), m.periodSlots)
	}
	codes := warningCodes(ws)
	if codes["unreachable-node"] != 0 || codes["expected-work"] != 1 {
		t.Errorf("warnings %v", ws)
	}
}

func warningCodes(ws []Warning) map[string]int {
	m := map[string]int{}
	for _, w := range ws {
		m[w.Code]++
	}
	return m
}

func TestCompileErrors(t *testing.T) {
	normal, _ := dist.NewNormal(0, 1)
	pareto1, _ := dist.NewPareto(1, 1)
	cat, _ := dist.NewCategorical([]float64{1, 2}, []float64{0.5, 0.5})
	cases := []struct {
		name string
		mod  func(g *Graph)
		want string
	}{
		{"duplicate node", func(g *Graph) { g.Nodes[1].ID = "a" }, "duplicate id"},
		{"empty node id", func(g *Graph) { g.Nodes[2].ID = "" }, "empty id"},
		{"source target", func(g *Graph) { g.Sources[0].To = "zz" }, "not found"},
		{"outcome target", func(g *Graph) { g.Nodes[0].Outcomes[0].To = "zz" }, "target node not found"},
		{"next target", func(g *Graph) { g.Nodes[1].Next = "zz" }, "not found"},
		{"outcomes and next", func(g *Graph) { g.Nodes[0].Next = "d" }, "both outcomes and next"},
		{"per-time on leaf", func(g *Graph) { g.Nodes[3].Functions[0].Basis = PerTime }, "per-time function on a leaf"},
		{"sum not 1", func(g *Graph) { g.Nodes[0].Outcomes[0].P = c(0.4) }, "sum to"},
		{"two remainders", func(g *Graph) { g.Nodes[0].Outcomes[0].P = nil; g.Nodes[0].Outcomes[1].P = nil }, "more than one outcome takes the remainder"},
		{"fixed exceeds 1", func(g *Graph) { g.Nodes[0].Outcomes[0].P = c(0.9); g.Nodes[0].Outcomes[1].P = c(0.9) }, "sum to"},
		{"fixed above 1", func(g *Graph) { g.Nodes[0].Outcomes[0].P = nil; g.Nodes[0].Outcomes[1].P = c(1.2) }, "does not fit [0, 1]"},
		{"uncertain with three", func(g *Graph) {
			g.Nodes[0].Outcomes = []Outcome{{To: "b", P: beta(2, 5)}, {To: "c", P: c(0.3)}, {To: "d"}}
		}, "sequential mode otherwise"},
		{"sequential last has p", func(g *Graph) {
			g.Nodes[0].Mode = Sequential
			g.Nodes[0].Outcomes = []Outcome{{To: "b", P: beta(2, 5)}, {To: "c", P: c(0.3)}}
		}, "the last outcome takes the remainder"},
		{"sequential missing p", func(g *Graph) {
			g.Nodes[0].Mode = Sequential
			g.Nodes[0].Outcomes = []Outcome{{To: "b"}, {To: "c"}}
		}, "needs a probability"},
		{"probability support", func(g *Graph) {
			g.Nodes[0].Outcomes = []Outcome{{To: "b", P: normal}, {To: "c"}}
		}, "does not fit [0, 1]"},
		{"holding support", func(g *Graph) { g.Nodes[0].Outcomes[0].Holding = normal }, "does not fit"},
		{"holding categorical ok", func(g *Graph) { g.Nodes[0].Outcomes[0].Holding = cat }, ""},
		{"frequency support", func(g *Graph) { g.Sources[0].Frequency = normal }, "does not fit"},
		{"infinite mean", func(g *Graph) { g.Nodes[3].Functions[0].Model = pareto1 }, "infinite mean"},
		{"cycle", func(g *Graph) { g.Nodes[3].Next = "a" }, "cycle: a -> b -> d -> a"},
		{"no sources", func(g *Graph) { g.Sources = nil }, "no sources"},
		{"no frequency", func(g *Graph) { g.Sources[0].Frequency = nil }, "no distribution"},
	}
	for _, tc := range cases {
		g := diamond()
		tc.mod(&g)
		_, _, err := g.Compile(Config{})
		if tc.want == "" {
			if err != nil {
				t.Errorf("%s: unexpected %v", tc.name, err)
			}
			continue
		}
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want %q", tc.name, err, tc.want)
		}
	}
	if _, _, err := diamond().Compile(Config{Watch: Watch{Nodes: []string{"zz"}}}); err == nil || !strings.Contains(err.Error(), "not a node") {
		t.Errorf("watch: %v", err)
	}
	if _, _, err := diamond().Compile(Config{Spill: -1}); err == nil {
		t.Error("negative spill accepted")
	}
}

func TestCompileSelections(t *testing.T) {
	g := diamond()
	// Two-way absolute: the uncertain outcome comes first.
	g.Nodes[0].Outcomes = []Outcome{{To: "b"}, {To: "c", P: beta(2, 5)}}
	m, _ := mustCompile(t, g, Config{})
	if !m.nodes[0].twoWay || m.OutcomeNames()[0] != "a→c" || m.outs[0].probPool == noIndex || m.outs[1].probPool != noIndex {
		t.Errorf("two-way: %+v %v", m.nodes[0], m.OutcomeNames())
	}
	if p := m.outcomeMeanProb(0, 0); math.Abs(p-2.0/7) > 1e-15 {
		t.Errorf("two-way mean prob %g", p)
	}
	// Sequential: P(j) = q_j Π(1 − q_i).
	g = diamond()
	g.Nodes[0].Mode = Sequential
	g.Nodes[0].Outcomes = []Outcome{{To: "b", P: c(0.5)}, {To: "c", P: beta(1, 1)}, {To: "d"}}
	m, _ = mustCompile(t, g, Config{})
	ps := []float64{m.outcomeMeanProb(0, 0), m.outcomeMeanProb(0, 1), m.outcomeMeanProb(0, 2)}
	if math.Abs(ps[0]-0.5) > 1e-15 || math.Abs(ps[1]-0.25) > 1e-15 || math.Abs(ps[2]-0.25) > 1e-15 {
		t.Errorf("sequential probabilities %v", ps)
	}
	// Wide fixed selection gets an alias table; repeated targets get suffixes.
	g = diamond()
	var outs []Outcome
	for i := 0; i < 10; i++ {
		outs = append(outs, Outcome{To: "d", P: c(0.1)})
	}
	g.Nodes[0].Outcomes = outs
	m, _ = mustCompile(t, g, Config{})
	if m.nodes[0].alias != 0 || len(m.aliasProb) != 10 || m.OutcomeNames()[1] != "a→d#2" {
		t.Errorf("alias %d probs %d names %v", m.nodes[0].alias, len(m.aliasProb), m.OutcomeNames()[:3])
	}
	sum := 0.0
	for i, p := range m.aliasProb {
		sum += p
		_ = m.aliasIdx[i]
	}
	if math.Abs(sum-10) > 1e-12 {
		t.Errorf("alias table mass %g", sum)
	}
}

func TestCompileSpillAndDelays(t *testing.T) {
	g := diamond()
	g.Nodes[0].Outcomes[0].Holding = c(2)
	g.Nodes[1].NextHolding = c(3)
	m, ws := mustCompile(t, g, Config{})
	// Longest weighted path: a→b (2) then b→d (3).
	if !m.HasDelays() || m.Spill() != 5 {
		t.Errorf("spill %d", m.Spill())
	}
	if codes := warningCodes(ws); codes["wide-window"] != 0 {
		t.Error("unexpected wide-window warning")
	}
	m, _ = mustCompile(t, g, Config{Spill: 7})
	if m.Spill() != 7 {
		t.Errorf("override spill %d", m.Spill())
	}
	// A heavy-tailed holding time drives S past the warning threshold.
	pa, _ := dist.NewPareto(1, 3)
	g.Nodes[1].NextHolding = pa
	m, ws = mustCompile(t, g, Config{})
	if m.Spill() <= spillWarn || warningCodes(ws)["wide-window"] != 1 {
		t.Errorf("pareto holding: spill %d warnings %v", m.Spill(), ws)
	}
	// A heavier tail puts S beyond any window: refused, not wrapped.
	pa, _ = dist.NewPareto(1, 1.5)
	g.Nodes[1].NextHolding = pa
	if _, _, err := g.Compile(Config{}); err == nil || !strings.Contains(err.Error(), "add a cap") {
		t.Errorf("pareto 1.5 holding: %v", err)
	}
}

func TestCompileUnreachable(t *testing.T) {
	g := diamond()
	g.Nodes = append(g.Nodes, Node{ID: "island"})
	_, ws := mustCompile(t, g, Config{})
	found := false
	for _, w := range ws {
		if w.Code == "unreachable-node" && w.Subject == "island" {
			found = true
		}
	}
	if !found {
		t.Errorf("warnings %v", ws)
	}
}

// TestReachCap: the reach tables are refused above the cap, with the
// counts named.
func TestReachCap(t *testing.T) {
	if err := reachCap("g", 10000, 10000); err == nil || !strings.Contains(err.Error(), "10000 × 10000") {
		t.Errorf("want a refusal: %v", err)
	}
	if err := reachCap("g", 100, 201); err != nil {
		t.Errorf("the register should fit: %v", err)
	}
}
