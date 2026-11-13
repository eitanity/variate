package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

const (
	// burnInShare is the largest share of a quantity's steady-state
	// value per period that the burn-in may leave out of period 0.
	burnInShare = 1e-4
	// maxBurnIn is the longest burn-in, in periods.
	maxBurnIn = 1<<31 - 1
	// burnInWarn is the burn-in above which Compile warns.
	burnInWarn = 1 << 20
	// pendSlack and pendFloor size the pending bookings: a multiple of the
	// expected number in flight at a block's end, plus a floor.
	pendSlack = 8
	pendFloor = 1024
)

// pathHolds is, per node, what the paths into it carry: the most holding
// times on any one of them, the outcomes with a holding time on any of
// them, and the most mean delay.
type pathHolds struct {
	n     []int      // the most holding times on a path into the node
	up    [][]uint64 // bit set of outcomes with a holding time on some path into the node
	delay []float64  // the largest sum of mean holding times on a path into the node
}

// pathHoldsOf propagates the holding times forward in topological order.
func (m *Compiled) pathHoldsOf() pathHolds {
	words := (len(m.outs) + 63) / 64
	ph := pathHolds{n: make([]int, len(m.nodes)), up: make([][]uint64, len(m.nodes)), delay: make([]float64, len(m.nodes))}
	for v := range m.nodes {
		ph.up[v] = make([]uint64, words)
	}
	for _, v := range m.topo {
		n := m.nodes[v]
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			o := m.outs[j]
			add, mean := 0, 0.0
			if o.hold != noIndex {
				add, mean = 1, m.pools[o.hold].Mean()
			}
			ph.n[o.to] = max(ph.n[o.to], ph.n[v]+add)
			ph.delay[o.to] = math.Max(ph.delay[o.to], ph.delay[v]+mean)
			for i, b := range ph.up[v] {
				ph.up[o.to][i] |= b
			}
			if o.hold != noIndex {
				ph.up[o.to][j/64] |= uint64(1) << (j % 64)
			}
		}
	}
	return ph
}

// burnInBound evaluates the share of value a burn-in of b periods leaves
// out of period 0, an upper bound over every node's entries and functions.
// A round that arrived before −b books into period 0 only what lands at
// least b periods after its arrival. With k holding times before a
// booking, its delay reaches b only if one of them reaches b/k, and the
// part of a rate's span past b is at most the sum of each holding time's
// excess over b/k. It returns the share and the holding outcome with the
// largest term.
func (m *Compiled) burnInBound(ph pathHolds, b float64) (share float64, driver int32) {
	tb := termBest{driver: noIndex, top: -1}
	// bound: nodes
	for v, n := range m.nodes {
		share = math.Max(share, m.entryShare(ph, v, b, &tb))
		if n.rateStart != n.rateEnd {
			share = math.Max(share, m.rateShare(ph, v, n, b, &tb))
		}
	}
	return share, tb.driver
}

// termBest tracks the holding outcome with the largest term.
type termBest struct {
	top    float64
	driver int32
}

func (tb *termBest) see(t float64, j int32) {
	if t > tb.top {
		tb.top, tb.driver = t, j
	}
}

// entryShare bounds the share of node v's entries that land b or more
// periods after their round's arrival.
func (m *Compiled) entryShare(ph pathHolds, v int, b float64, tb *termBest) float64 {
	k := ph.n[v]
	if k == 0 {
		return 0
	}
	s := 0.0
	m.eachUp(ph, v, func(j int32) {
		t := m.pools[m.outs[j].hold].Survival(b / float64(k))
		s += t
		tb.see(t, j)
	})
	return s
}

// rateShare bounds, over the outcomes of v with a holding time, the share
// of a rate's value spread over that holding time that lands b or more
// periods after the round's arrival.
func (m *Compiled) rateShare(ph pathHolds, v int, n cnode, b float64, tb *termBest) float64 {
	share := 0.0
	c := b / float64(ph.n[v]+1)
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		o := m.outs[j]
		if o.hold == noIndex {
			continue
		}
		mean := m.pools[o.hold].Mean()
		if !(mean > 0) {
			continue
		}
		s := dist.ExcessMean(m.pools[o.hold], c)
		tb.see(s, j)
		m.eachUp(ph, v, func(i int32) {
			t := dist.ExcessMean(m.pools[m.outs[i].hold], c)
			s += t
			tb.see(t, i)
		})
		share = math.Max(share, s/mean)
	}
	return share
}

// eachUp calls f for each holding outcome on some path into node v.
func (m *Compiled) eachUp(ph pathHolds, v int, f func(j int32)) {
	for i, word := range ph.up[v] {
		// bound: 64 bits of a word
		for bit := 0; bit < 64 && word != 0; bit++ {
			if word&1 != 0 {
				f(int32(i*64 + bit)) //nolint:gosec // an outcome index
			}
			word >>= 1
		}
	}
}

// setBurnIn computes the burn-in B, the shortest whose bound is at most
// burnInShare, or takes the override, and sizes the pending bookings.
func (m *Compiled) setBurnIn(override int, ws *[]Warning) error {
	if override < 0 {
		return modelError("config", "BurnIn", "must be >= 0")
	}
	if !m.hasDelays {
		m.burnIn = 0
		return nil
	}
	ph := m.pathHoldsOf()
	m.pendCap = m.pendingCap(ph)
	if override > 0 {
		m.burnIn = int64(override)
		return nil
	}
	share, driver := m.burnInBound(ph, maxBurnIn)
	if share > burnInShare || math.IsNaN(share) {
		return modelError("outcome", m.outcomeNames[driver], "a burn-in of "+itoa(maxBurnIn)+" periods would still leave "+figure(share)+" of a period's value out, above "+ftoa(burnInShare)+"; add a cap")
	}
	lo, hi := int64(0), int64(maxBurnIn)
	// bound: 31 halvings of [0, 2^31)
	for hi-lo > 1 {
		mid := lo + (hi-lo)/2
		if s, _ := m.burnInBound(ph, float64(mid)); s <= burnInShare {
			hi = mid
		} else {
			lo = mid
		}
	}
	if s, _ := m.burnInBound(ph, 0); s <= burnInShare {
		hi = 0
	}
	m.burnIn = hi
	if hi > burnInWarn {
		_, driver = m.burnInBound(ph, float64(hi))
		*ws = append(*ws, Warning{Code: "long-burn-in", Subject: m.outcomeNames[driver], Message: "burn-in of " + itoa(int(hi)) + " periods, run once for each of the " + itoa(batchGroups) + " independent timelines that measure the error; a cap on this holding time would shorten it"})
	}
	return nil
}

// pendingCap bounds the bookings in flight past a block's end: per
// source, its rounds per period times, per node it reaches, the mean
// delay into the node times the columns an entry books, and for each
// rate the delay to the end of the longest holding out of the node, by
// Little's law; then pendSlack times that, plus pendFloor.
func (m *Compiled) pendingCap(ph pathHolds) int {
	flight := 0.0
	for k := range m.srcs {
		lam := m.countMean[k]
		for v := range m.nodes {
			if r := m.reach[k][v]; r > 0 {
				flight += lam * r * m.nodeFlight(ph, v)
			}
		}
	}
	if !(flight < 1<<26) {
		flight = 1 << 26
	}
	return int(math.Ceil(pendSlack*flight)) + pendFloor
}

// nodeFlight is what one entry into node v leaves in flight past a
// block's end, as the mean delay times the bookings: the columns its
// entry books, and a segment per rate, which lasts to the end of the
// longest holding out of v.
func (m *Compiled) nodeFlight(ph pathHolds, v int) float64 {
	n := m.nodes[v]
	cols := 1 + 4*float64(n.fnEnd-n.fnStart)
	if n.thresh != noIndex {
		cols += 4
	}
	hold := 0.0
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		o := m.outs[j]
		if o.hold != noIndex {
			hold = math.Max(hold, m.pools[o.hold].Mean())
		}
		if o.fn != noIndex {
			cols += 4
		}
	}
	return ph.delay[v]*cols + (ph.delay[v]+hold)*float64(n.rateEnd-n.rateStart)
}
