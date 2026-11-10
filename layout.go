package variate

// layout maps the per-period quantities onto one row of A values:
// total, categories, sources, entries per node, contribution per node,
// the layers of the total, and the sum each trigger tests.
type layout struct {
	cats, srcs, nodes, layers, trigs int
	width                            int
	// The first row of each block, so that a booking adds one offset.
	srcAt, entryAt, contribAt, layerAt, trigAt int
}

func newLayout(m *Compiled) layout {
	l := layout{cats: len(m.categories), srcs: len(m.srcs), nodes: len(m.nodes), layers: len(m.layers), trigs: len(m.trigs)}
	l.srcAt = 1 + l.cats
	l.entryAt = l.srcAt + l.srcs
	l.contribAt = l.entryAt + l.nodes
	l.layerAt = l.contribAt + l.nodes
	l.trigAt = l.layerAt + l.layers
	l.width = l.trigAt + l.trigs
	return l
}

func (l *layout) layer(j int) int { return l.layerAt + j }
func (l *layout) trig(i int) int  { return l.trigAt + i }

func (l *layout) total() int          { return 0 }
func (l *layout) tailWidth() int      { return l.srcAt + l.srcs } // the rows the tail shares sum
func (l *layout) cat(c int32) int     { return 1 + int(c) }
func (l *layout) src(k int32) int     { return l.srcAt + int(k) }
func (l *layout) entry(v int32) int   { return l.entryAt + int(v) }
func (l *layout) contrib(v int32) int { return l.contribAt + int(v) }
