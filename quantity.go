package variate

// Quantities a run keeps a histogram of, beyond the total: each
// category, each source, and each node's entries and contribution when
// Config.NodeHist asks for them. These accessors let a report draw any
// of them the way it draws the total.

// The kinds of quantity a row breaks the total into, as QuantityRow
// names them.
const (
	kindCategory = "category"
	kindSource   = "source"
	kindEntries  = "entries"
	kindContrib  = "contrib"
)

// QuantityRow finds the row of a quantity by kind ("total", "category",
// "source", "entries", "contrib", "layer", "trigger") and name; ok is false for a kind or
// name the model does not have, and for a breakdown with Config.NoBreakdown.
func (w *Workspace) QuantityRow(kind, name string) (row int, ok bool) {
	return w.m.quantityRow(w.l, kind, name)
}

func (m *Compiled) quantityRow(l layout, kind, name string) (row int, ok bool) {
	if kind == totalSubject {
		return l.total(), true
	}
	names, at := m.quantityNames(l, kind)
	for i, n := range names {
		if n == name {
			return at(i), true
		}
	}
	return 0, false
}

// quantityNames lists the names of a kind and the row of each.
func (m *Compiled) quantityNames(l layout, kind string) (names []string, at func(int) int) {
	if !m.crit.breakdown && (kind == kindCategory || kind == kindSource || kind == kindContrib) {
		return nil, nil
	}
	switch kind {
	case kindCategory:
		return m.categories, func(i int) int { return l.cat(index(i)) }
	case kindSource:
		return m.sourceIDs, func(i int) int { return l.src(index(i)) }
	case kindEntries:
		return m.nodeIDs, func(i int) int { return l.entry(index(i)) }
	case kindContrib:
		return m.nodeIDs, func(i int) int { return l.contrib(index(i)) }
	case "layer":
		names = make([]string, len(m.layers))
		for i, y := range m.layers {
			names[i] = y.ID
		}
		return names, l.layer
	case "trigger":
		names = make([]string, len(m.trigs))
		for i, tr := range m.trigs {
			names[i] = m.sourceIDs[tr.src]
		}
		return names, l.trig
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

// StructureSource is a source and the node it enters. A trigger is a
// source with the nodes whose bookings its sum adds up, and its level.
type StructureSource struct {
	ID, To  string
	Trigger bool
	Inputs  []string
	Over    float64
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
		if n.thresh != noIndex {
			f := n.thresh
			sn.Functions = append(sn.Functions, StructureFunction{Name: m.fnNames[f], Category: m.categories[m.fns[f].category]})
		}
		s.Nodes = append(s.Nodes, sn)
	}
	for j, o := range m.outs {
		s.Outcomes = append(s.Outcomes, StructureEdge{Name: m.outcomeNames[j], From: m.nodeIDs[m.outcomeNode[j]], To: m.nodeIDs[o.to], Holding: o.hold != noIndex})
	}
	for k, src := range m.srcs {
		ss := StructureSource{ID: m.sourceIDs[k], To: m.nodeIDs[src.to]}
		if src.trigger != noIndex {
			ss.Trigger, ss.Over = true, m.trigs[src.trigger].over
			ss.Inputs = m.triggerInputs(src.trigger)
		}
		s.Sources = append(s.Sources, ss)
	}
	return s
}
