package config

import "code.eitanity.com/variate"

// parseDependence reads the groups of values that move together:
// dependence: [ { functions: [node.function, node->to], rank: r } ].
func (l *loader) parseDependence(top map[string]node) error {
	x, ok := top["dependence"]
	if !ok {
		return nil
	}
	items, err := x.items()
	if err != nil {
		return err
	}
	for _, it := range items {
		m, err := it.mapping("functions", "rank")
		if err != nil {
			return err
		}
		f, ok := m["functions"]
		r, okRank := m["rank"]
		if !ok || !okRank {
			return it.errorf("dependence needs functions and rank")
		}
		var d variate.Dependence
		if d.Members, err = names(f); err != nil {
			return err
		}
		if d.Rank, err = r.float(); err != nil {
			return err
		}
		l.model.Graph.Dependence = append(l.model.Graph.Dependence, d)
	}
	return nil
}
