package config

import (
	"math"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// Block is a reliability diagram a source stands on: the stationary
// frequency of the system's failures, its MTBF and its unavailability,
// for independent components with exponential failure and repair
// times. The engine simulates the diagram's units exactly; these
// figures are its long-run mean.
type Block struct {
	Source         string
	Rate           float64 // failures per period
	MTBF           float64 // hours between failures, downtime included
	Unavailability float64 // long-run share of time down
}

// The keys a block map may hold: one of the three forms, a repair time
// with the component form, and a time unit.
const (
	blockLeaf   = "mtbf"
	blockKofn   = "kofn"
	blockSeries = "series"
	maxUnits    = 64
)

// blockFrame is one block of the diagram during parsing: its chains
// once known, and the series it belongs to.
type blockFrame struct {
	parent int // index of the enclosing series frame, or -1
	series bool
	chains []variate.Chain
}

// pendingBlock is a block map still to be read, with the time unit it
// inherits and the series frame it belongs to.
type pendingBlock struct {
	x      node
	unit   string
	parent int
}

// parseBlock reads a block map into the chains of its diagram in
// series: a component is a chain of one unit, k of n a chain, and a
// series the chains of its parts, nested series flattening since the
// series form is associative. Nested series are walked with an explicit
// stack: every block becomes a frame after its parent, so a pass from
// the last frame back gathers each series from its parts.
func (l *loader) parseBlock(x node, unit string) ([]variate.Chain, error) {
	stack := []pendingBlock{{x, unit, -1}}
	var frames []blockFrame
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		frame, parts, err := l.readBlock(p)
		if err != nil {
			return nil, err
		}
		stack = append(stack, parts...)
		for i := range parts {
			stack[len(stack)-1-i].parent = len(frames)
		}
		frames = append(frames, frame)
	}
	return gatherSeries(frames), nil
}

// readBlock reads one block map into a frame: a component or k-of-n
// block has its chain at once; a series returns its parts to be read,
// with the unit in force.
func (l *loader) readBlock(p pendingBlock) (blockFrame, []pendingBlock, error) {
	form, value, m, unit, err := l.blockForm(p.x, p.unit)
	if err != nil {
		return blockFrame{}, nil, err
	}
	frame := blockFrame{parent: p.parent}
	switch form {
	case blockLeaf:
		ch, err := l.leafChain(p.x, value, m["mttr"], unit, 1, 1)
		frame.chains = []variate.Chain{ch}
		return frame, nil, err
	case blockKofn:
		ch, err := l.kofnChain(value, unit)
		frame.chains = []variate.Chain{ch}
		return frame, nil, err
	default:
		items, err := value.items()
		if err != nil {
			return blockFrame{}, nil, err
		}
		if len(items) == 0 {
			return blockFrame{}, nil, value.errorf("series needs at least one block")
		}
		frame.series = true
		parts := make([]pendingBlock, len(items))
		for i, it := range items {
			parts[i] = pendingBlock{x: it, unit: unit}
		}
		return frame, parts, nil
	}
}

// gatherSeries fills each series frame with its parts' chains, last
// frame first so that a nested series is complete before its parent
// reads it, in the order the parts were written, and returns the root's.
func gatherSeries(frames []blockFrame) []variate.Chain {
	for i := len(frames) - 1; i >= 0; i-- {
		if !frames[i].series {
			continue
		}
		var parts [][]variate.Chain
		for j := i + 1; j < len(frames); j++ {
			if frames[j].parent == i {
				parts = append(parts, frames[j].chains)
			}
		}
		// Parts were pushed in order and popped in reverse, so they sit
		// in reverse order of writing.
		for j := len(parts) - 1; j >= 0; j-- {
			frames[i].chains = append(frames[i].chains, parts[j]...)
		}
	}
	return frames[0].chains
}

// blockForm reads a block map's keys: which of the three forms it is,
// that form's value, the whole map, and the time unit in force.
func (l *loader) blockForm(x node, unit string) (form string, value node, m map[string]node, u string, err error) {
	if m, err = x.mapping(blockLeaf, "mttr", blockKofn, blockSeries, "unit"); err != nil {
		return "", node{}, nil, "", err
	}
	u = unit
	if n, ok := m["unit"]; ok {
		if u, err = n.oneOf("hours", "days", "weeks"); err != nil {
			return "", node{}, nil, "", err
		}
	}
	for _, k := range []string{blockLeaf, blockKofn, blockSeries} {
		if v, ok := m[k]; ok {
			if form != "" {
				return "", node{}, nil, "", x.errorf("a block is one of mtbf, kofn or series")
			}
			form, value = k, v
		}
	}
	if form == "" {
		return "", node{}, nil, "", x.errorf("a block is one of mtbf, kofn or series")
	}
	if _, ok := m["mttr"]; ok && form != blockLeaf {
		return "", node{}, nil, "", x.errorf("mttr goes with mtbf")
	}
	return form, value, m, u, nil
}

// leafChain reads a component's mean up time and, optionally, its mean
// repair time, into a chain of n units of which k must work, with the
// rates per period; no repair time means replaced at once.
func (l *loader) leafChain(x, mtbf, mttr node, unit string, k, n int) (variate.Chain, error) {
	if unit == "" {
		return variate.Chain{}, x.errorf("a block needs unit: hours, days or weeks")
	}
	up, err := mtbf.float()
	if err != nil {
		return variate.Chain{}, err
	}
	if !(up > 0) || math.IsInf(up, 1) {
		return variate.Chain{}, mtbf.errorf("mtbf must be positive")
	}
	ch := variate.Chain{K: k, N: n, Fail: l.periodHours / (up * unitHours(unit))}
	if mttr.n != nil {
		down, err := mttr.float()
		if err != nil {
			return variate.Chain{}, err
		}
		if down < 0 || math.IsInf(down, 1) || math.IsNaN(down) {
			return variate.Chain{}, mttr.errorf("mttr must be zero or positive")
		}
		if down > 0 {
			ch.Repair = l.periodHours / (down * unitHours(unit))
		}
	}
	return ch, nil
}

// kofnChain reads n identical components of which k must be working.
func (l *loader) kofnChain(x node, unit string) (variate.Chain, error) {
	m, err := x.mapping("k", "n", blockLeaf, "mttr", "unit")
	if err != nil {
		return variate.Chain{}, err
	}
	if u, ok := m["unit"]; ok {
		if unit, err = u.oneOf("hours", "days", "weeks"); err != nil {
			return variate.Chain{}, err
		}
	}
	if m["k"].n == nil || m["n"].n == nil || m[blockLeaf].n == nil {
		return variate.Chain{}, x.errorf("kofn needs k, n and mtbf")
	}
	n, err := m["n"].integer(1, maxUnits)
	if err != nil {
		return variate.Chain{}, err
	}
	k, err := m["k"].integer(1, n)
	if err != nil {
		return variate.Chain{}, err
	}
	return l.leafChain(x, m[blockLeaf], m["mttr"], unit, int(k), int(n))
}

// blockSource reads a source's block. The engine simulates the chains
// exactly when any unit has a repair time; when every unit is replaced
// at once the diagram is a Poisson source at its stationary rate, with
// exact variance. The block's figures are recorded on the model.
func (l *loader) blockSource(id string, x node, s *variate.Source) error {
	if l.periodHours == 0 {
		return x.errorf("a block needs period: declared")
	}
	chains, err := l.parseBlock(x, "")
	if err != nil {
		return err
	}
	rate, u := variate.ChainFigures(chains)
	l.model.Blocks = append(l.model.Blocks, Block{Source: id, Rate: rate, MTBF: l.periodHours / rate, Unavailability: u})
	repaired := false
	for _, ch := range chains {
		repaired = repaired || ch.Repair > 0
	}
	if !repaired {
		d, err := dist.NewConstant(rate)
		if err != nil {
			return x.errorf("block rate: %v", err)
		}
		s.Frequency = d
		return nil
	}
	s.Chains = chains
	return nil
}
