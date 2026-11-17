package variate

import (
	"math"
	"slices"

	"code.eitanity.com/variate/stats"
)

// Total is a sum of categories the model names, less those in Less,
// reported beside the period total as a quantity of its own.
type Total struct {
	ID         string
	Categories []string
	Less       []string // subtracted: such a total can be negative
}

// ctotal is a compiled named total: which categories it sums, and which
// of them it subtracts.
type ctotal struct {
	id  string
	has []bool // [category]
	neg []bool // [category]: subtracted; nil when none is
}

// sign is +1 for a category the total adds, -1 for one it subtracts and
// 0 for one it leaves out.
func (t *ctotal) sign(c int) float64 {
	switch {
	case !t.has[c]:
		return 0
	case t.neg != nil && t.neg[c]:
		return -1
	}
	return 1
}

// signed reports whether the total subtracts a category.
func (t *ctotal) signed() bool { return t.neg != nil }

// flattenTotals compiles the named totals and the categories kept out
// of the total. Every category named must be one a function books.
func (m *Compiled) flattenTotals(totals []Total, outside []string, cats map[string]int32) error {
	if err := m.flattenOutside(outside, cats); err != nil {
		return err
	}
	seen := map[string]bool{}
	for i, t := range totals {
		if t.ID == "" || seen[t.ID] {
			return modelError("total", itoa(i), "empty or duplicate id")
		}
		seen[t.ID] = true
		ct, err := compileTotal(t, cats)
		if err != nil {
			return err
		}
		m.totals = append(m.totals, ct)
		if !ct.signed() {
			m.gridTotals = append(m.gridTotals, index(i))
		}
	}
	return nil
}

// flattenOutside marks the categories kept out of the total; at least
// one must stay in it.
func (m *Compiled) flattenOutside(outside []string, cats map[string]int32) error {
	m.inTotal = make([]bool, len(cats))
	for c := range m.inTotal {
		m.inTotal[c] = true
	}
	for _, name := range outside {
		c, ok := cats[name]
		if !ok {
			return modelError("outside_total", name, "no function books this category")
		}
		m.inTotal[c] = false
	}
	if len(cats) > 0 && m.countInTotal() == 0 {
		return modelError("outside_total", outside[0], "every category is outside the total")
	}
	return nil
}

// compileTotal compiles one named total: the categories it adds, and
// those it subtracts.
func compileTotal(t Total, cats map[string]int32) (ctotal, error) {
	if len(t.Categories) == 0 {
		return ctotal{}, modelError("total", t.ID, "no categories")
	}
	ct := ctotal{id: t.ID, has: make([]bool, len(cats))}
	if err := ct.add(t.ID, t.Categories, cats); err != nil {
		return ctotal{}, err
	}
	if len(t.Less) == 0 {
		return ct, nil
	}
	added := slices.Clone(ct.has)
	if err := ct.add(t.ID, t.Less, cats); err != nil {
		return ctotal{}, err
	}
	ct.neg = make([]bool, len(cats))
	for c := range ct.neg {
		ct.neg[c] = ct.has[c] && !added[c]
	}
	return ct, nil
}

// add marks the named categories, each one a function books and none
// marked already.
func (t *ctotal) add(id string, names []string, cats map[string]int32) error {
	for _, name := range names {
		c, ok := cats[name]
		switch {
		case !ok:
			return modelError("total", id, "no function books category "+name)
		case t.has[c]:
			return modelError("total", id, "category "+name+" is listed twice")
		}
		t.has[c] = true
	}
	return nil
}

func (m *Compiled) countInTotal() int {
	n := 0
	for _, in := range m.inTotal {
		if in {
			n++
		}
	}
	return n
}

// fnInTotal reports whether function f books into the total.
func (m *Compiled) fnInTotal(f int32) bool { return m.inTotal[m.fns[f].category] }

// TotalIDs returns the named totals' ids in the model's order.
func (m *Compiled) TotalIDs() []string {
	out := make([]string, len(m.totals))
	for i, t := range m.totals {
		out[i] = t.id
	}
	return out
}

// InTotal reports whether category c is booked into the total.
func (m *Compiled) InTotal(c int) bool { return m.inTotal[c] }

// totalMoments sets each named total's exact mean, the sum of its
// categories' means, and its variance from a pass over its functions.
func (m *Compiled) totalMoments(e *Expectation, lam []float64) {
	m.totalMeans(e)
	e.VarByTotal = make([]float64, len(m.totals))
	for i := range m.totals {
		if m.Started() {
			e.VarByTotal[i] = math.NaN()
			continue
		}
		f := valueFilter{kind: filterTotal, index: index(i)}
		t1, t2 := m.valueToGo(f)
		e.VarByTotal[i] = m.variance(f, lam, t1, t2, e.MeanByTotal[i])
	}
}

// totalMeans sets each named total's mean, the sum of its categories'.
func (m *Compiled) totalMeans(e *Expectation) {
	e.MeanByTotal = make([]float64, len(m.totals))
	for i := range m.totals {
		t := &m.totals[i]
		for c, in := range t.has {
			if in {
				e.MeanByTotal[i] += t.sign(c) * e.MeanByCategory[c]
			}
		}
	}
}

// selects reports whether function f books into what the transform's
// index cat stands for: -1 the total, a category, or from the number of
// categories on, a named total that only adds, in gridTotals' order.
func (m *Compiled) selects(f, cat int32) bool {
	c := m.fns[f].category
	switch n := index(len(m.categories)); {
	case cat == -1:
		return m.inTotal[c]
	case cat < n:
		return c == cat
	default:
		return m.totals[m.gridTotals[cat-n]].has[c]
	}
}

// sumTotals sets each named total's column of row to the sum of its
// categories' columns.
func (m *Compiled) sumTotals(row []float64, l layout) {
	for i := range m.totals {
		t := &m.totals[i]
		s := 0.0
		for c, in := range t.has {
			if in {
				s += t.sign(c) * row[l.cat(index(c))]
			}
		}
		row[l.named(i)] = s
	}
}

// allocTotals gives each named total its accumulator and a histogram
// ranged from its exact mean; one that subtracts takes a signed
// histogram, ranged from its categories' means added whatever their
// sign.
func (w *Workspace) allocTotals(e *Expectation, res float64) {
	r := &w.result
	r.Totals = make([]Accum, len(w.m.totals))
	for i := range w.m.totals {
		t := &w.m.totals[i]
		q := w.l.named(i)
		w.means[q] = e.MeanByTotal[i]
		if t.signed() {
			scale := 0.0
			for c, in := range t.has {
				if in {
					scale += math.Abs(e.MeanByCategory[c])
				}
			}
			r.Totals[i].Hist = stats.NewSignedHist(scale, res)
		} else {
			r.Totals[i].Hist = stats.NewLogHist(e.MeanByTotal[i], res)
		}
		w.hists[q] = r.Totals[i].Hist
	}
}

// sharesTotal reports whether the quantity at row is a part of the
// total, so that its share of it means something: a category in the
// total, a source, a node's contribution, or a named total that adds
// categories in the total.
func (w *Workspace) sharesTotal(row int) bool {
	l := &w.l
	switch {
	case row >= 1 && row < l.srcAt:
		return w.m.inTotal[row-1]
	case row >= l.srcAt && row < l.entryAt, row >= l.contribAt && row < l.totalAt:
		return true
	case row >= l.totalAt && row < l.layerAt:
		t := w.m.totals[row-l.totalAt]
		if t.signed() {
			return false
		}
		for c, in := range t.has {
			if in && !w.m.inTotal[c] {
				return false
			}
		}
		return true
	}
	return false
}

// shareBase is what a share of the quantity at row is taken of: the
// total's mean, or NaN for a category or named total outside it.
func (w *Workspace) shareBase(row int) float64 {
	l := &w.l
	if row >= l.totalAt && row < l.layerAt || row >= 1 && row < l.srcAt {
		if !w.sharesTotal(row) {
			return math.NaN()
		}
	}
	return w.result.Total.Mean()
}
