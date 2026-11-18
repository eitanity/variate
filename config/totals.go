package config

import "code.eitanity.com/variate"

// parseTotals reads the named totals, each an id, the categories it
// sums and those it subtracts, the categories kept outside the total,
// and the discount.
func (l *loader) parseTotals(top map[string]node) error {
	if x, ok := top["totals"]; ok {
		items, err := x.items()
		if err != nil {
			return err
		}
		for _, it := range items {
			t, err := parseTotal(it)
			if err != nil {
				return err
			}
			l.model.Graph.Totals = append(l.model.Graph.Totals, t)
		}
	}
	if x, ok := top["outside_total"]; ok {
		out, err := names(x)
		if err != nil {
			return err
		}
		l.model.Graph.OutsideTotal = out
	}
	return l.parseDiscount(top)
}

// parseTotal reads one named total.
func parseTotal(it node) (variate.Total, error) {
	m, err := it.mapping("id", "description", "categories", "less")
	if err != nil {
		return variate.Total{}, err
	}
	t := variate.Total{}
	if t.ID, err = required(it, m, "id"); err != nil {
		return t, err
	}
	c, ok := m["categories"]
	if !ok {
		return t, it.errorf("total %s has no categories", t.ID)
	}
	if t.Categories, err = names(c); err != nil {
		return t, err
	}
	if x, ok := m["less"]; ok {
		if t.Less, err = names(x); err != nil {
			return t, err
		}
	}
	return t, nil
}

// names reads a list of names.
func names(x node) ([]string, error) {
	items, err := x.items()
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(items))
	for _, it := range items {
		s, err := it.str()
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// parseDiscount reads discount: { rate | factors, categories }, a
// started run's discount on the categories it names.
func (l *loader) parseDiscount(top map[string]node) error {
	x, ok := top["discount"]
	if !ok {
		return nil
	}
	m, err := x.mapping("rate", "factors", "categories")
	if err != nil {
		return err
	}
	d := &l.model.Graph.Discount
	c, ok := m["categories"]
	if !ok {
		return x.errorf("discount names no categories")
	}
	if d.Categories, err = names(c); err != nil {
		return err
	}
	if r, ok := m["rate"]; ok {
		if d.Rate, err = r.float(); err != nil {
			return err
		}
	}
	if f, ok := m["factors"]; ok {
		if d.Factors, err = f.floats(0); err != nil {
			return err
		}
	}
	if _, ok := m["rate"]; !ok && d.Factors == nil {
		return x.errorf("discount needs a rate or factors")
	}
	return nil
}
