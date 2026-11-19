package variate

import (
	"math"
	"slices"
	"testing"

	"code.eitanity.com/variate/dist"
)

// ranks is each value's rank in xs.
func ranks(xs []float64) []float64 {
	idx := make([]int, len(xs))
	for i := range idx {
		idx[i] = i
	}
	slices.SortFunc(idx, func(a, b int) int {
		switch {
		case xs[a] < xs[b]:
			return -1
		case xs[a] > xs[b]:
			return 1
		}
		return 0
	})
	out := make([]float64, len(xs))
	for r, i := range idx {
		out[i] = float64(r)
	}
	return out
}

// pearson is the correlation of two samples.
func pearson(x, y []float64) float64 {
	n := float64(len(x))
	var sx, sy, sxx, syy, sxy float64
	for i := range x {
		sx, sy, sxx, syy, sxy = sx+x[i], sy+y[i], sxx+x[i]*x[i], syy+y[i]*y[i], sxy+x[i]*y[i]
	}
	return (sxy/n - sx*sy/n/n) / math.Sqrt((sxx/n-sx*sx/n/n)*(syy/n-sy*sy/n/n))
}

// TestTogetherDraws: two members drawn by the copula keep their laws, a
// gamma's and a PERT's quantiles within their errors, and have the
// stated rank correlation within its error; each round draws the
// group's normal once.
func TestTogetherDraws(t *testing.T) {
	x, _ := dist.NewGammaFromMeanSD(2, 1)
	y, _ := dist.NewPERT(1, 3, 10, 4)
	for _, rank := range []float64{0.3, 0.8, 1} {
		g := Graph{ID: "t",
			Sources:    []Source{{ID: "s", Frequency: dist.Constant{}, To: "a"}},
			Nodes:      []Node{{ID: "a", Functions: []Function{fn("x", "k", x), fn("y", "k", y)}}},
			Dependence: []Dependence{{Members: []string{"a.x", "a.y"}, Rank: rank}},
		}
		m := compiled(t, g)
		w := &Workspace{m: m}
		wk := newWorker(m)
		wk.k.seed = 7
		const n = 200000
		xs, ys := make([]float64, n), make([]float64, n)
		for i := range n {
			wk.rkey = splitmix(uint64(i))
			xs[i] = w.together(wk, 0, m.fns[0].pool)
			ys[i] = w.together(wk, 0, m.fns[1].pool)
		}
		got := pearson(ranks(xs), ranks(ys))
		if se := (1 - rank*rank) / math.Sqrt(n); math.Abs(got-rank) > 4*se+1e-4 {
			t.Errorf("rank %v: %v", rank, got)
		}
		for _, c := range []struct {
			s []float64
			d dist.Distribution
		}{{xs, x}, {ys, y}} {
			slices.Sort(c.s)
			for _, p := range []float64{0.1, 0.5, 0.9, 0.99} {
				q := c.s[int(p*n)]
				if dp := c.d.CDF(q) - p; math.Abs(dp) > 4*math.Sqrt(p*(1-p)/n) {
					t.Errorf("rank %v: P(X <= %v) off by %v at %v", rank, q, dp, p)
				}
			}
		}
	}
}
