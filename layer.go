package variate

import "math"

// Layer is a function of the period total reported beside it: the part
// of the total between an attachment and the attachment plus a limit,
// times a share, as an aggregate deductible, an aggregate limit or a
// reinsurance layer takes it. A layer is computed from each closed
// period's total, so it has the statistics, histogram, series and
// quantity access of any other row; when the transform applies, its
// mean has exact bounds from the total's bracket.
type Layer struct {
	ID         string
	Attachment float64 // the total below which the layer pays nothing
	Limit      float64 // the most the layer pays before the share; 0 = no limit
	Share      float64 // the share of the band the layer pays; 0 = all of it
}

// payout is what the layer pays on a period total.
func (y Layer) payout(total float64) float64 {
	return y.Share * math.Min(math.Max(total-y.Attachment, 0), y.Limit)
}

// normalised fills the defaults: no limit, the whole band.
func (y Layer) normalised() Layer {
	if y.Limit <= 0 {
		y.Limit = math.Inf(1)
	}
	if y.Share <= 0 {
		y.Share = 1
	}
	return y
}

// Layers are the model's layers, with their defaults filled in.
func (m *Compiled) Layers() []Layer { return m.layers }

// applyLayers fills a closed period's layer rows from its total.
func (w *Workspace) applyLayers(row []float64, total float64) {
	for j, y := range w.m.layers {
		row[w.l.layer(j)] = y.payout(total)
	}
}

// LayerMean brackets E[payout] of a layer from a bracket on P(T ≥ x):
// the payout is the share times the integral of P(T ≥ t) over the
// band, and on each lattice cell the lower curve at the cell's end
// and the upper curve at its start bound the integrand.
func (b Bounds) LayerMean(y Layer) (lo, hi float64) {
	y = y.normalised()
	end := y.Attachment + y.Limit
	// bound: lattice points
	for k := 0; k < len(b.Lower); k++ {
		x0 := float64(k) * b.H
		x1 := x0 + b.H
		from := math.Max(x0, y.Attachment)
		to := math.Min(x1, end)
		if to <= from {
			if x0 >= end {
				break
			}
			continue
		}
		width := to - from
		if k+1 < len(b.Lower) {
			lo += width * b.Lower[k+1]
		}
		hi += width * b.Upper[k]
	}
	if math.IsInf(end, 1) && len(b.Upper) > 0 {
		// Beyond the lattice the upper curve is not known to fall; the
		// bracket's last value stands over the remaining band only when
		// it is zero, else the upper bound is open.
		if b.Upper[len(b.Upper)-1] > 0 {
			hi = math.Inf(1)
		}
	}
	return y.Share * lo, y.Share * hi
}

// LayerBoundsAt reads a bracket on the layer's payout at x from the
// total's bracket: the payout reaches x when the total reaches the
// attachment plus x over the share, and never exceeds the share times
// the limit.
func LayerBoundsAt(b Bounds, y Layer, x float64) (lo, hi float64) {
	y = y.normalised()
	if x > y.Share*y.Limit {
		return 0, 0
	}
	t := y.Attachment + x/y.Share
	k := min(int(math.Ceil(t/b.H)), len(b.Lower)-1)
	return b.Lower[k], b.Upper[k]
}
