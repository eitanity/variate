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
	return nil
}

// Levels returns Config.At as compiled.
func (m *Compiled) Levels() []Level {
	out := make([]Level, len(m.levels))
	for i, lv := range m.levels {
		out[i] = lv.Level
	}
	return out
}
