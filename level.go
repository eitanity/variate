package variate

import "math"

// clevel is one of Config.At compiled: the row of its quantity and its
// value.
type clevel struct {
	Level
	row int
	x   float64
}

// compileLevels finds the row of each of Config.At's quantities; a
// quantity the model does not have, one the breakdown leaves out, or a
// value that is not a finite number is an error.
func (m *Compiled) compileLevels(at []Level) error {
	l := newLayout(m)
	m.levels = make([]clevel, 0, len(at))
	for _, lv := range at {
		subject := lv.Kind
		if lv.Name != "" {
			subject += ":" + lv.Name
		}
		row, ok := m.quantityRow(l, lv.Kind, lv.Name)
		switch {
		case !ok:
			return modelError("config", "At", "no quantity "+subject+"; name one as kind:name, as category:labour or trigger:overrun, or total")
		case math.IsNaN(lv.X) || math.IsInf(lv.X, 0):
			return modelError("config", "At", subject+": the value must be a finite number")
		}
		m.levels = append(m.levels, clevel{Level: lv, row: row, x: lv.X})
	}
	m.userLevels = len(m.levels)
	// A named total that subtracts counts the periods it is above 0, so
	// that P(Q ≤ 0) is counted as the levels are.
	for i := range m.totals {
		if t := &m.totals[i]; t.signed() {
			lv := Level{Kind: totalSubject, Name: t.id, X: math.SmallestNonzeroFloat64}
			m.levels = append(m.levels, clevel{Level: lv, row: l.named(i), x: lv.X})
		}
	}
	return nil
}

// Levels returns Config.At as compiled.
func (m *Compiled) Levels() []Level {
	out := make([]Level, m.userLevels)
	for i, lv := range m.levels[:m.userLevels] {
		out[i] = lv.Level
	}
	return out
}
