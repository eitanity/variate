package variate

func newSlot(l layout, s int, srcs, nodes, outs, thresholds int) *slot {
	sl := &slot{
		window:     make([]float64, (s+1)*l.width),
		head:       make([]float64, s*l.width),
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
