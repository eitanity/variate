package variate

func newSlot(l layout, block int64, pendCap, nslots, srcs, nodes, outs, thresholds int) *slot {
	sl := &slot{
		rows:       make([]float64, int(block)*l.width),
		far:        make([]float64, l.width),
		pend:       newPending(pendCap),
		incoming:   make([]float64, nslots+1),
		stats:      make([]qstat, l.width),
		events:     make([]uint64, srcs),
		reach:      make([]uint64, srcs*nodes),
		outs:       make([]uint64, srcs*outs),
		tailCounts: make([]int64, thresholds),
		tailSums:   make([]float64, thresholds*l.tailWidth()),
	}
	sl.reset()
	return sl
}
