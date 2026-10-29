package variate

import (
	"math"
	"os"

	"code.eitanity.com/variate/stats"
)

// qstat is one quantity's statistics within a slot, merged into the
// global Accum at finalisation.
type qstat struct {
	sum      stats.Neumaier
	w        stats.Welford
	min, max float64
	zero     int64
}

func (q *qstat) reset() {
	*q = qstat{min: math.Inf(1), max: math.Inf(-1)}
}

func (q *qstat) add(x float64) {
	q.sum.Add(x)
	q.w.Add(x)
	if x < q.min {
		q.min = x
	}
	if x > q.max {
		q.max = x
	}
	if x == 0 {
		q.zero++
	}
}

// slot is the working state of one task: the window of S+1 rows, the head
// of S rows awaiting the previous task's spill, the slot statistics, and
// the task's integer counts.
type slot struct {
	id     uint32 // the task's stream key half
	task   int64
	a, b   int64 // the block [a, b)
	window []float64
	head   []float64
	stats  []qstat
	events []uint64
	reach  []uint64 // [source×nodes]
	outs   []uint64 // [source×outcomes]
	steps  uint64
	base   int   // ring row of the period being simulated
	closed int64 // periods closed by the task (excluding the head)
	tail   int64 // closed periods with Total at or beyond the tail quantile bound
	// Per tail threshold, the periods that reached it and their rows summed.
	tailCounts []int64
	tailSums   []float64 // [threshold×width]
	err        error
	// Round stream: the block's matching rounds, in delivery order, in the
	// buffer and then in the file.
	recBuf  []byte
	recLen  int
	recFile *os.File
	fileLen int64
	records int64
}

func (sl *slot) reset() {
	clear(sl.window)
	clear(sl.head)
	for i := range sl.stats {
		sl.stats[i].reset()
	}
	clear(sl.events)
	clear(sl.reach)
	clear(sl.outs)
	clear(sl.tailCounts)
	clear(sl.tailSums)
	sl.steps = 0
	sl.closed = 0
	sl.tail = 0
	sl.err = nil
	sl.recLen = 0
	sl.fileLen = 0
	sl.records = 0
}

// row returns the window row of period q for a task whose earliest period
// is first.
func (sl *slot) row(l layout, s int, first, q int64) []float64 {
	r := int((q - first) % int64(s+1))
	return sl.window[r*l.width : (r+1)*l.width]
}
