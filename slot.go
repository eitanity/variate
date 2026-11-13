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

// slot is the working state of one task: a row per period of its block,
// the bookings past the block, the slot statistics, and the task's
// integer counts.
type slot struct {
	id       uint32 // the task's stream key half
	task     int64
	a, b     int64     // the block [a, b)
	rows     []float64 // a row per period of [a, b)
	far      []float64 // scratch row for a booking past the block
	farAt    int64     // the period far holds, or below a when it holds none
	pend     pending   // the bookings past the block
	incoming []float64 // what the finaliser handed in, by distance back to the task that booked it
	stats    []qstat
	coarse   []*stats.LogHist // with delays, the block's group histogram per quantity; nil entries without one
	events   []uint64
	reach    []uint64 // [source×nodes]
	outs     []uint64 // [source×outcomes]
	steps    uint64
	closed   int64 // periods closed by the task
	tail     int64 // closed periods with Total at or beyond the tail quantile bound
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
	clear(sl.rows)
	clear(sl.far)
	sl.farAt = math.MinInt64
	sl.pend.reset()
	clear(sl.incoming)
	for i := range sl.stats {
		sl.stats[i].reset()
	}
	for _, h := range sl.coarse {
		if h != nil {
			h.Reset()
		}
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

// rowAt returns the row of period q, within the block.
func (sl *slot) rowAt(l layout, q int64) []float64 {
	r := int(q - sl.a)
	return sl.rows[r*l.width : (r+1)*l.width]
}
