package variate

import (
	"math"

	"code.eitanity.com/variate/stats"
)

// Total is a sum of categories the model names, reported beside the
// period total as a quantity of its own.
type Total struct {
	ID         string
	Categories []string
}

// ctotal is a compiled named total: which categories it sums.
type ctotal struct {
	id  string
	has []bool // [category]
}

// flattenTotals compiles the named totals and the categories kept out
// of the total. Every category named must be one a function books.
func (m *Compiled) flattenTotals(totals []Total, outside []string, cats map[string]int32) error {
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
	seen := map[string]bool{}
	for i, t := range totals {
		if t.ID == "" || seen[t.ID] {
			return modelError("total", itoa(i), "empty or duplicate id")
		}
		seen[t.ID] = true
		if len(t.Categories) == 0 {
			return modelError("total", t.ID, "no categories")
		}
		ct := ctotal{id: t.ID, has: make([]bool, len(cats))}
		for _, name := range t.Categories {
			c, ok := cats[name]
			switch {
			case !ok:
				return modelError("total", t.ID, "no function books category "+name)
			case ct.has[c]:
				return modelError("total", t.ID, "category "+name+" is listed twice")
			}
			ct.has[c] = true
		}
		m.totals = append(m.totals, ct)
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
	for i, t := range m.totals {
		for c, in := range t.has {
			if in {
				e.MeanByTotal[i] += e.MeanByCategory[c]
			}
		}
	}
}

// selects reports whether function f books into what the transform's
// index cat stands for: -1 the total, a category, or from the number of
// categories on, a named total.
func (m *Compiled) selects(f, cat int32) bool {
	c := m.fns[f].category
	switch n := index(len(m.categories)); {
	case cat == -1:
		return m.inTotal[c]
	case cat < n:
		return c == cat
	default:
		return m.totals[cat-n].has[c]
	}
}

// sumTotals sets each named total's column of row to the sum of its
// categories' columns.
func (m *Compiled) sumTotals(row []float64, l layout) {
	for i, t := range m.totals {
		s := 0.0
		for c, in := range t.has {
			if in {
				s += row[l.cat(index(c))]
			}
		}
		row[l.named(i)] = s
	}
}

// allocTotals gives each named total its accumulator and a histogram
// ranged from its exact mean.
func (w *Workspace) allocTotals(e *Expectation, res float64) {
	r := &w.result
	r.Totals = make([]Accum, len(w.m.totals))
	for i := range w.m.totals {
		q := w.l.named(i)
		w.means[q] = e.MeanByTotal[i]
		r.Totals[i].Hist = stats.NewLogHist(e.MeanByTotal[i], res)
		w.hists[q] = r.Totals[i].Hist
	}
}

// sharesTotal reports whether the quantity at row is a part of the
// total, so that its share of it means something: a category in the
// total, a source, a node's contribution, or a named total of
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
