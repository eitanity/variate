package config

import "code.eitanity.com/variate"

// parseTotals reads the named totals, each an id and the categories it
// sums, and the categories kept outside the total.
func (l *loader) parseTotals(top map[string]node) error {
	if x, ok := top["totals"]; ok {
		items, err := x.items()
		if err != nil {
			return err
		}
		for _, it := range items {
			m, err := it.mapping("id", "description", "categories")
			if err != nil {
				return err
			}
			t := variate.Total{}
			if t.ID, err = required(it, m, "id"); err != nil {
				return err
			}
			c, ok := m["categories"]
			if !ok {
				return it.errorf("total %s has no categories", t.ID)
			}
			if t.Categories, err = names(c); err != nil {
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
	return nil
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
