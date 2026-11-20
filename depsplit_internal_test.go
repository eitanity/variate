package variate

import (
	"math"
	"testing"

	"code.eitanity.com/variate/dist"
)

// heldGroup is a round that books x, then after the holding tau y, with
// a lost rate over the holding, at a gamma rate of mean 8 and sd 2: x,
// the holding and y move together at rank.
func heldGroup(t *testing.T, tau dist.Distribution, rank float64) *Compiled {
	t.Helper()
	x, _ := dist.NewGammaFromMeanSD(3, 1)
	y, _ := dist.NewGammaFromMeanSD(5, 2)
	four, _ := dist.NewConstant(4)
	rate, _ := dist.NewGammaFromMeanSD(8, 2)
	g := Graph{ID: "held",
		Sources: []Source{{ID: "s", Frequency: rate, To: "a"}},
		Nodes: []Node{
			{ID: "a", Functions: []Function{{Name: "x", Category: "x", Model: x, Basis: OnEntry}, {Name: "r", Category: "lost", Model: four, Basis: PerTime}}, Next: "b", NextHolding: tau},
			{ID: "b", Functions: []Function{{Name: "y", Category: "y", Model: y, Basis: OnEntry}}},
		},
		Dependence: []Dependence{{Members: []string{"a.x", "a->b", "b.y"}, Rank: rank}},
	}
	m, _, err := g.Compile(Config{})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestDepBetweenSettled: the rule over the group's normal is chosen by
// the within-period figures, and the between-period term on it is that
// of a finer rule to 1e-12 of the figure: a Gauss–Hermite rule of twice
// the nodes over 2 days at rank 0.9 and a year at rank 0.5, and over 3
// months at rank 0.95 a split rule of 64 nodes a piece, at least twice
// the chosen rule's on every piece.
func TestDepBetweenSettled(t *testing.T) {
	cases := []struct {
		med, rank float64
		nodes     int
		split     bool
	}{{2.0 / 365, 0.9, 128, false}, {1, 0.5, 64, false}, {0.25, 0.95, 64, true}}
	for _, c := range cases {
		tau, _ := dist.NewLognormalFromMedianP90(c.med, 3*c.med)
		m := heldGroup(t, tau, c.rank)
		keys, lam := m.delayKeys(), m.countMean
		cs, cw, start, ok := m.depRule(keys, lam)
		if !ok {
			t.Fatalf("median %v rank %v: no rule", c.med, c.rank)
		}
		within := settleGrids(start, cs, cw, keys, lam)
		bs, _, _, ok := m.ownBetween(cs, cw, keys, lam, within)
		if !ok {
			t.Fatalf("median %v rank %v: no between-period term", c.med, c.rank)
		}
		var xs, ws []float64
		if kinks, _ := m.groupKinks(0, 0); c.split {
			xs, ws = splitRule(kinks, c.nodes)
		} else {
			xs, ws = make([]float64, c.nodes), make([]float64, c.nodes)
			dist.HermiteRule(xs, ws)
		}
		fine, fw, _ := m.depPoints([][]float64{xs}, [][]float64{ws}, &scoreCache{z: map[[2]int][2][]float64{}})
		ref, _, _, _ := m.ownBetween(fine, fw, keys, lam, settleGrids(start, fine, fw, keys, lam))
		for k := range keys {
			if d := math.Abs(bs[k]-ref[k]) / math.Abs(within[k]+bs[k]); !(d <= 1e-12) {
				t.Errorf("median %v rank %v key %d: between %v, finer %v", c.med, c.rank, k, bs[k], ref[k])
			}
		}
	}
}

// TestLockstepRule: a point holding's rule, spread over a cell, from the
// uniform's moments in closed form, is the rule on its sub-cells' Gauss
// points to 1e-11, below 0, inside a cell, on a cell's edge, about the
// period's end and beyond it.
func TestLockstepRule(t *testing.T) {
	const n = 256
	h := 1.0 / n
	for _, tau := range []float64{0.001, 0.3, 0.30390625, 0.999, 1.0005, 2} {
		u, _ := dist.NewUniform(tau-h/2, tau+h/2)
		a := lockstepRule(tau, n)
		b := lagRuleWith(u, n, func(mu *[4][]float64) (float64, float64) { return subMoments(u, n, mu) })
		worst := math.Max(math.Abs(a.f0-b.f0), math.Max(math.Abs(a.tail-b.tail), math.Abs(a.eG2-b.eG2)))
		for i := range a.c {
			worst = math.Max(worst, math.Abs(a.c[i]-b.c[i]))
		}
		for i := range a.eH {
			worst = math.Max(worst, math.Abs(a.eH[i]-b.eH[i]))
		}
		if !(worst <= 1e-11) {
			t.Errorf("holding %v: rules %.2g apart", tau, worst)
		}
	}
}

// TestDelayMemo: a value factor scaling a value that moves with a
// holding, its delay figures read from the quadratic through three of
// the factor's points once a fourth fits it, gives the figures of every
// point computed, to 1e-9, at rank 0.5 and in lockstep.
func TestDelayMemo(t *testing.T) {
	x, _ := dist.NewGammaFromMeanSD(3, 1)
	y, _ := dist.NewGammaFromMeanSD(5, 2)
	f, _ := dist.NewLognormalFromMeanSD(1, 0.3)
	eight, _ := dist.NewConstant(8)
	tau, _ := dist.NewLognormalFromMedianP90(0.25, 0.75)
	for _, rank := range []float64{0.5, 1} {
		g := Graph{ID: "memo",
			Factors: []Factor{{ID: "f", Model: f}},
			Sources: []Source{{ID: "s", Frequency: eight, To: "a"}},
			Nodes: []Node{
				{ID: "a", Functions: []Function{{Name: "x", Category: "x", Model: x, Basis: OnEntry, Factor: "f"}}, Next: "b", NextHolding: tau},
				{ID: "b", Functions: []Function{{Name: "y", Category: "y", Model: y, Basis: OnEntry}}},
			},
			Dependence: []Dependence{{Members: []string{"a.x", "a->b", "b.y"}, Rank: rank}},
		}
		m, _, err := g.Compile(Config{})
		if err != nil {
			t.Fatal(err)
		}
		run := func(memo bool) *Expectation {
			rules := newRuleCache()
			if !memo {
				rules.delays.quad = -1
			}
			var e *Expectation
			for _, n := range []int{valueNodesStart, 2 * valueNodesStart} {
				var ok bool
				if e, ok, err = m.mixedAnalytic(m.valueFactors(), n, rules); err != nil || !ok {
					t.Fatalf("rank %v: %v", rank, err)
				}
			}
			if memo && rules.delays.quad != 1 {
				t.Errorf("rank %v: the quadratic did not hold", rank)
			}
			return e
		}
		got, want := run(true), run(false)
		figs := append([]float64{got.Var}, got.VarByCategory...)
		wants := append([]float64{want.Var}, want.VarByCategory...)
		for i := range figs {
			if !(math.Abs(figs[i]-wants[i]) <= 1e-9*math.Abs(wants[i])) {
				t.Errorf("rank %v figure %d: %v, every point %v", rank, i, figs[i], wants[i])
			}
		}
	}
}
