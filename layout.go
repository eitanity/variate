package variate

// layout maps the per-period quantities onto one row of A values:
// total, categories, sources, entries per node, contribution per node,
// the layers of the total, the sum each trigger tests, the queues'
// columns, and 1 or 0 for each of Config.At as its value is reached.
type layout struct {
	cats, srcs, nodes, layers, trigs, queues, levels int
	width                                            int
	// The first row of each block, so that a booking adds one offset.
	srcAt, entryAt, contribAt, layerAt, trigAt, queueAt, levelAt int
}

func newLayout(m *Compiled) layout {
	l := layout{cats: len(m.categories), srcs: len(m.srcs), nodes: len(m.nodes), layers: len(m.layers), trigs: len(m.trigs), queues: len(m.queues), levels: len(m.levels)}
	l.srcAt = 1 + l.cats
	l.entryAt = l.srcAt + l.srcs
	l.contribAt = l.entryAt + l.nodes
	l.layerAt = l.contribAt + l.nodes
	l.trigAt = l.layerAt + l.layers
	l.queueAt = l.trigAt + l.trigs
	l.levelAt = l.queueAt + queueCols*l.queues
	l.width = l.levelAt + l.levels
	return l
}

func (l *layout) layer(j int) int { return l.layerAt + j }
func (l *layout) trig(i int) int  { return l.trigAt + i }

// queue is queue c's column col, one of the qJoined … qUnfinished.
func (l *layout) queue(c, col int) int { return l.queueAt + queueCols*c + col }

// level is the column counting whether the i-th of Config.At is reached.
func (l *layout) level(i int) int { return l.levelAt + i }

func (l *layout) total() int          { return 0 }
func (l *layout) cat(c int32) int     { return 1 + int(c) }
func (l *layout) src(k int32) int     { return l.srcAt + int(k) }
func (l *layout) entry(v int32) int   { return l.entryAt + int(v) }
func (l *layout) contrib(v int32) int { return l.contribAt + int(v) }
