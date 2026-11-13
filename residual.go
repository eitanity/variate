package variate

import (
	"math"
	"math/rand/v2"

	"code.eitanity.com/variate/dist"
)

const (
	// residualCellsPerOctave sets the envelope's resolution.
	residualCellsPerOctave = 8
	// residualFloor is the first cell's end, as a share of the mean.
	residualFloor = 0x1p-40
	// residualTop is where the table ends, in periods: the mass beyond
	// it is an atom there, as no run is that long.
	residualTop = 0x1p63
	// residualRejections bounds the proposals of one draw.
	residualRejections = 1000
)

// residTable draws the equilibrium residual of a repair time h, with
// density P(h > r)/E[h], by rejection under a step envelope. The
// survival function never rises, so its value at a cell's left edge
// bounds it across the cell.
type residTable struct {
	d     dist.Distribution
	edges []float64 // cell i is [edges[i], edges[i+1]); the last cell is the atom at residualTop
	env   []float64 // the survival function at each cell's left edge
	prob  []float64 // alias table over the cells' envelope mass
	alias []int32
}

// newResidTable builds the table for a repair time with a finite mean.
func newResidTable(d dist.Distribution) residTable {
	mean := d.Mean()
	_, hi := d.Support()
	top := math.Min(hi, residualTop)
	step := math.Exp2(1.0 / residualCellsPerOctave)
	edges := []float64{0}
	// bound: octaves from mean·2^−40 to 2^63, a few hundred cells each way
	for e := mean * residualFloor; e < top; e *= step {
		edges = append(edges, e)
	}
	edges = append(edges, top)
	n := len(edges) - 1
	t := residTable{d: d, edges: edges, env: make([]float64, n)}
	mass := make([]float64, n+1)
	total := 0.0
	// bound: the cells
	for i := 0; i < n; i++ {
		t.env[i] = d.Survival(edges[i])
		mass[i] = t.env[i] * (edges[i+1] - edges[i])
		total += mass[i]
	}
	if top < hi {
		mass[n] = dist.ExcessMean(d, top)
		total += mass[n]
	}
	for i := range mass {
		mass[i] /= total
	}
	t.prob, t.alias = buildAlias(mass)
	return t
}

// draw takes a residual with three uniforms per proposal. Survival is
// called once per proposal, at a block's start only.
func (t *residTable) draw(r *rand.Rand) float64 {
	n := len(t.env)
	x := 0.0
	// bound: residualRejections proposals, after which the last stands
	for range residualRejections {
		s := r.Float64() * float64(len(t.prob))
		i := int(s)
		if s-float64(i) >= t.prob[i] {
			i = int(t.alias[i])
		}
		if i == n {
			return t.edges[n]
		}
		x = t.edges[i] + r.Float64()*(t.edges[i+1]-t.edges[i])
		if r.Float64()*t.env[i] < t.d.Survival(x) {
			return x
		}
	}
	return x
}

// buildResiduals gives every source with a repair time that is not a
// constant its residual table.
func (m *Compiled) buildResiduals() {
	m.resid = make([]residTable, len(m.srcs))
	for k, s := range m.srcs {
		if s.repair == noIndex {
			continue
		}
		if lo, hi := m.pools[s.repair].Support(); lo != hi {
			m.resid[k] = newResidTable(m.pools[s.repair])
		}
	}
}
