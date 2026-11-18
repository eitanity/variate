package variate

import "math"

// Discount weighs what a started run books in chosen categories by the
// period it lands in, so their figures are present values. Period p of
// the horizon, p from 1, has the factor (1+Rate)^-p, the end of the
// period, or Factors[p-1] in place of Rate for any other curve.
type Discount struct {
	Rate       float64   // per period; > -1
	Factors    []float64 // one per period of the horizon, each >= 0; nil = from Rate
	Categories []string  // the categories discounted; nil = no discount
}

// flattenBooks compiles how the categories are booked: the named
// totals, those kept out of the total, and the discount.
func (b *builder) flattenBooks(m *Compiled, cats map[string]int32) error {
	if err := m.flattenTotals(b.g.Totals, b.g.OutsideTotal, cats); err != nil {
		return err
	}
	return m.flattenDiscount(b.g.Discount, b.g.Start.Horizon, cats)
}

// flattenDiscount compiles the discount: which categories it applies
// to, and each period's factor. A discount whose every factor is 1 is
// none, so it changes nothing.
func (m *Compiled) flattenDiscount(d Discount, horizon int, cats map[string]int32) error {
	m.discounted = make([]bool, len(cats))
	if d.Categories == nil && d.Factors == nil && d.Rate == 0 {
		return nil
	}
	factors, err := discountFactors(d, horizon)
	if err != nil {
		return err
	}
	if err := m.markDiscounted(d.Categories, cats); err != nil {
		return err
	}
	// bound: the horizon
	for _, f := range factors {
		if f != 1 {
			m.discount = factors
			return nil
		}
	}
	clear(m.discounted)
	return nil
}

// discountFactors is each period's factor, from the rate or as given.
func discountFactors(d Discount, horizon int) ([]float64, error) {
	switch {
	case horizon == 0:
		return nil, modelError("discount", "", "weighs a started run's periods; a steady-state run has no first period")
	case len(d.Categories) == 0:
		return nil, modelError("discount", "", "names no categories")
	case d.Factors != nil && d.Rate != 0:
		return nil, modelError("discount", "", "has both a rate and factors")
	case d.Factors != nil && len(d.Factors) != horizon:
		return nil, modelError("discount", "factors", "has "+itoa(len(d.Factors))+" factors for a horizon of "+itoa(horizon)+" periods")
	case d.Factors == nil && !(d.Rate > -1 && d.Rate < math.Inf(1)):
		return nil, modelError("discount", "rate", "must be greater than -1 and finite")
	}
	if d.Factors != nil {
		// bound: the horizon
		for p, f := range d.Factors {
			if !(f >= 0 && f < math.Inf(1)) {
				return nil, modelError("discount", "factors", "period "+itoa(p+1)+": a factor must be >= 0 and finite")
			}
		}
		return append([]float64(nil), d.Factors...), nil
	}
	out := make([]float64, horizon)
	v := 1 / (1 + d.Rate)
	f := 1.0
	// bound: the horizon
	for p := range out {
		f *= v
		out[p] = f
	}
	return out, nil
}

// markDiscounted marks the categories discounted. Each must be one a
// function books, named once; a named total is discounted through its
// categories, so naming one is refused.
func (m *Compiled) markDiscounted(names []string, cats map[string]int32) error {
	// bound: the categories named
	for _, name := range names {
		c, ok := cats[name]
		switch {
		case ok && m.discounted[c]:
			return modelError("discount", name, "category is listed twice")
		case ok:
			m.discounted[c] = true
		case m.totalIndex(name) >= 0:
			return modelError("discount", name, "is a total, discounted through its categories: list those")
		default:
			return modelError("discount", name, "no function books this category")
		}
	}
	return nil
}

// totalIndex is the named total's place, or -1.
func (m *Compiled) totalIndex(id string) int {
	for i := range m.totals {
		if m.totals[i].id == id {
			return i
		}
	}
	return -1
}

// Discounted reports whether the run discounts any category.
func (m *Compiled) Discounted() bool { return m.discount != nil }

// DiscountFactors is each period's factor, nil without a discount.
func (m *Compiled) DiscountFactors() []float64 { return append([]float64(nil), m.discount...) }

// DiscountedCategories names the categories discounted.
func (m *Compiled) DiscountedCategories() []string {
	var out []string
	for c, ok := range m.discounted {
		if ok && m.discount != nil {
			out = append(out, m.categories[c])
		}
	}
	return out
}

// factor is period q's discount factor: 1 without a discount, or past
// the horizon, where a started run drops what it books.
func (w *Workspace) factor(q int64) float64 {
	if q < 0 || q >= int64(len(w.m.discount)) {
		return 1
	}
	return w.m.discount[q]
}
