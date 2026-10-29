package variate

import "math"

// Quantities a run keeps a histogram of, beyond the total: each
// category, each source, and each node's entries and contribution when
// Config.NodeHist asks for them. These accessors let a report draw any
// of them the way it draws the total.

// QuantityRow finds the row of a quantity by kind ("total", "category",
// "source", "entries", "contrib", "layer") and name; ok is false for a kind or
// name the model does not have.
func (w *Workspace) QuantityRow(kind, name string) (row int, ok bool) {
	l := w.l
	if kind == "total" {
		return l.total(), true
	}
	names, at := w.quantityNames(kind)
	for i, n := range names {
		if n == name {
			return at(i), true
		}
	}
	return 0, false
}

// quantityNames lists the names of a kind and the row of each.
func (w *Workspace) quantityNames(kind string) (names []string, at func(int) int) {
	l := w.l
	switch kind {
	case "category":
		return w.m.categories, func(i int) int { return l.cat(index(i)) }
	case "source":
		return w.m.sourceIDs, func(i int) int { return l.src(index(i)) }
	case "entries":
		return w.m.nodeIDs, func(i int) int { return l.entry(index(i)) }
	case "contrib":
		return w.m.nodeIDs, func(i int) int { return l.contrib(index(i)) }
	case "layer":
		names = make([]string, len(w.m.layers))
		for i, y := range w.m.layers {
			names[i] = y.ID
		}
		return names, l.layer
	}
	return nil, nil
}

// HasHistogram reports whether a quantity's row keeps a histogram.
func (w *Workspace) HasHistogram(row int) bool {
	return row >= 0 && row < len(w.hists) && w.hists[row] != nil
}

// BandsOf fills dst[i] with P(edges[i] ≤ Q < edges[i+1]) for the
// quantity at row; dst must have len(edges)−1 entries. A row without a
// histogram leaves dst zero.
func (w *Workspace) BandsOf(row int, edges, dst []float64) []float64 {
	if w.HasHistogram(row) {
		w.hists[row].Bands(edges, dst)
	}
	return dst
}

// ExceedanceOf returns log-spaced points of P(Q ≥ x) for the quantity
// at row, as Summary.Exceedance does for the total.
func (w *Workspace) ExceedanceOf(row int, dst []ECPoint) []ECPoint {
	if !w.HasHistogram(row) {
		return dst
	}
	return w.exceedanceOf(w.accum(row), dst)
}

// QuantityStat is the statistics of the quantity at row, with its
// share of the total for a category or source.
func (w *Workspace) QuantityStat(row int) Stat {
	total := 1.0
	if (row > 0 && row < 1+w.l.cats+w.l.srcs) || row >= 1+w.l.cats+w.l.srcs+2*w.l.nodes {
		total = w.result.Total.Mean()
	}
	return w.stat(w.accum(row), total)
}

// TailShare is the share of the total each source and category booked
// over the periods whose total reached X: who the tail belongs to.
type TailShare struct {
	X          float64
	Periods    int64
	BySource   map[string]float64
	ByCategory map[string]float64
}

// tailShares computes the shares at every threshold with a period.
func (w *Workspace) tailShares(dst []TailShare) []TailShare {
	r := &w.result
	l := w.l
	dst = dst[:0]
	for j, x := range w.m.tailThresholds {
		if r.TailCounts[j] == 0 {
			continue
		}
		sums := r.TailSums[j]
		total := sums[l.total()]
		ts := TailShare{X: x, Periods: r.TailCounts[j], BySource: map[string]float64{}, ByCategory: map[string]float64{}}
		for k, name := range r.Sources {
			ts.BySource[name] = share(sums[l.src(index(k))], total)
		}
		for c, name := range r.Categories {
			ts.ByCategory[name] = share(sums[l.cat(index(c))], total)
		}
		dst = append(dst, ts)
	}
	return dst
}

func share(part, total float64) float64 {
	if total == 0 {
		return math.NaN()
	}
	return part / total
}

// addTail adds a closed period's total, categories and sources to the
// sums of every threshold its total reached; the thresholds ascend, so
// the first miss ends it.
func (w *Workspace) addTail(sl *slot, row []float64, total float64) {
	width := w.l.tailWidth()
	// bound: tail thresholds
	for j, x := range w.m.tailThresholds {
		if total < x {
			return
		}
		sl.tailCounts[j]++
		sums := sl.tailSums[j*width : (j+1)*width]
		for i := range sums {
			sums[i] += row[i]
		}
	}
}

// Thresholds of the tail shares: powers of two about the exact mean,
// from a sixteenth to 2^19 of it, fixed at compile so that every period
// is counted against the same levels.
const (
	tailThresholdCount = 24
	tailThresholdFirst = -4
)

// tailLevels is the thresholds for a mean, or none for a mean that is
// not positive and finite.
func tailLevels(mean float64) []float64 {
	if !(mean > 0) || math.IsInf(mean, 1) {
		return nil
	}
	out := make([]float64, tailThresholdCount)
	for j := range out {
		out[j] = mean * math.Pow(2, float64(j+tailThresholdFirst))
	}
	return out
}

// Structure describes the compiled graph for drawing: the nodes with
// their functions, the outcomes as edges, and the sources with the
// node each enters.
type Structure struct {
	Nodes    []StructureNode
	Outcomes []StructureEdge
	Sources  []StructureSource
}

// StructureNode is a node and the functions it books.
type StructureNode struct {
	ID        string
	Functions []StructureFunction
	Leaf      bool
}

// StructureFunction is a function's name, category and basis.
type StructureFunction struct {
	Name, Category string
	PerTime        bool
}

// StructureEdge is an outcome: from a node to its target, named as the
// report names it, holding when a holding time delays it.
type StructureEdge struct {
	Name, From, To string
	Holding        bool
}

// StructureSource is a source and the node it enters.
type StructureSource struct {
	ID, To string
}

// Structure returns the graph's drawing description.
func (m *Compiled) Structure() Structure {
	s := Structure{}
	for v, n := range m.nodes {
		sn := StructureNode{ID: m.nodeIDs[v], Leaf: n.outStart == n.outEnd}
		// bound: functions of one node
		for f := n.fnStart; f < n.rateEnd; f++ {
			sn.Functions = append(sn.Functions, StructureFunction{Name: m.fnNames[f], Category: m.categories[m.fns[f].category], PerTime: m.fnPerTime[f]})
		}
		s.Nodes = append(s.Nodes, sn)
	}
	for j, o := range m.outs {
		s.Outcomes = append(s.Outcomes, StructureEdge{Name: m.outcomeNames[j], From: m.nodeIDs[m.outcomeNode[j]], To: m.nodeIDs[o.to], Holding: o.hold != noIndex})
	}
	for k, src := range m.srcs {
		s.Sources = append(s.Sources, StructureSource{ID: m.sourceIDs[k], To: m.nodeIDs[src.to]})
	}
	return s
}
