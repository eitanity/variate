package variate

import "math"

// Chain is one block of a reliability diagram the engine simulates
// exactly: N identical units of which K must be working, each failing
// at rate Fail while up and repaired at rate Repair while down, both per
// period. A Repair of 0 means a unit is replaced at once. A source's
// chains are in series: the source is up when every chain has at least
// K working units, and it fires when that stops being so.
type Chain struct {
	K, N         int
	Fail, Repair float64
}

// The stationary figures of chains in series, from the frequency and
// unavailability algebra of independent exponential units: a chain's
// failed count is binomial with the unit's unavailability, the chain
// fails when, with exactly N − K down, one of the K working units
// fails, and the series fails when a chain fails while the others are
// up.

// ChainFigures returns the stationary failure frequency per period of
// a source's chains in series and the share of time it is down.
func ChainFigures(chains []Chain) (rate, unavailability float64) {
	up := 1.0
	for _, c := range chains {
		_, u := c.figures()
		up *= 1 - u
	}
	for _, c := range chains {
		f, u := c.figures()
		if u < 1 {
			rate += f * up / (1 - u)
		}
	}
	return rate, 1 - up
}

// figures is one chain's failure frequency and unavailability.
func (c Chain) figures() (f, u float64) {
	p := c.unitDown()
	q := 1 - p
	// bound: the chain's units
	for j := c.N - c.K + 1; j <= c.N; j++ {
		u += binomialCoeff(c.N, j) * math.Pow(p, float64(j)) * math.Pow(q, float64(c.N-j))
	}
	f = float64(c.K) * c.Fail * binomialCoeff(c.N, c.N-c.K) * math.Pow(p, float64(c.N-c.K)) * math.Pow(q, float64(c.K))
	return f, u
}

// unitDown is the share of time one unit is down.
func (c Chain) unitDown() float64 {
	if c.Repair <= 0 {
		return 0
	}
	return c.Fail / (c.Fail + c.Repair)
}

func binomialCoeff(n, j int) float64 {
	out := 1.0
	// bound: j, at most the units of a chain
	for i := 1; i <= j; i++ {
		out *= float64(n-j+i) / float64(i)
	}
	return out
}

// chainsUp reports whether every chain of the source has at least K
// working units, from the worker's failed counts.
func (w *Workspace) chainsUp(wk *worker, src csrc) bool {
	// bound: the source's chains
	for c := src.chainStart; c < src.chainEnd; c++ {
		ch := w.m.chains[c]
		if int(wk.failed[c]) > ch.N-ch.K {
			return false
		}
	}
	return true
}

// startChains puts a source's chains in their stationary state: each
// unit down with its unavailability, from the control stream.
func (w *Workspace) startChains(wk *worker, src csrc) {
	// bound: the source's chains
	for c := src.chainStart; c < src.chainEnd; c++ {
		ch := w.m.chains[c]
		p := ch.unitDown()
		wk.failed[c] = 0
		// bound: the chain's units
		for i := 0; i < ch.N; i++ {
			if wk.ss.Control().Float64() < p {
				wk.failed[c]++
			}
		}
	}
}

// runChainSource simulates the period's unit failures and repairs of a
// source's chains as one Markov chain on the failed counts, firing the
// source when the system goes from up to down. A unit replaced at once
// never stays down, so its chain fires the system directly when all its
// units are needed and never otherwise. When the system goes down the
// chains are run on until it is up again, so the round carries its
// outage for a holding that asks for it, and the period resumes from
// there; a look-ahead past the period's end is where the next period
// starts.
func (w *Workspace) runChainSource(wk *worker, sl *slot, k int32, p, first int64) {
	m := w.m
	src := m.srcs[k]
	end := float64(p) + 1
	t := math.Max(float64(p), wk.chainT[k])
	wk.chainT[k] = end
	up := w.chainsUp(wk, src)
	drawn := 0
	// bound: the period's unit events, a Poisson count at the total rate, capped
	for sl.err == nil {
		if drawn++; drawn > maxEventsPeriod {
			sl.err = eventsCapError(m.sourceIDs[k], p)
			return
		}
		total := 0.0
		// bound: the source's chains
		for c := src.chainStart; c < src.chainEnd; c++ {
			ch := m.chains[c]
			total += float64(ch.N-int(wk.failed[c]))*ch.Fail + float64(wk.failed[c])*ch.Repair
		}
		if !(total > 0) {
			return
		}
		t += wk.ss.Control().ExpFloat64() / total
		if t >= end {
			return
		}
		fired := w.chainEvent(wk, src, wk.ss.Control().Float64()*total)
		nowUp := w.chainsUp(wk, src)
		if (up && !nowUp) || (fired && up) {
			t = w.fireChain(wk, sl, k, src, t, p, first, nowUp)
			nowUp = true
		}
		up = nowUp
	}
}

// fireChain walks the round a system failure at t starts, with the
// outage the chains take to bring the system up when it is down, and
// returns the time the period resumes from: the end of the outage, or
// t itself for a unit replaced at once.
func (w *Workspace) fireChain(wk *worker, sl *slot, k int32, src csrc, t float64, p, first int64, nowUp bool) float64 {
	wk.outage = 0
	resume := t
	if !nowUp {
		resume = w.lookAhead(wk, src, t, float64(p)+float64(w.s)+1)
		wk.outage = resume - t
		if resume > float64(p)+1 {
			wk.chainT[k] = resume
		}
	}
	if p >= 0 {
		sl.events[k]++
	}
	if w.stream {
		w.beginRound(wk, t)
	}
	w.walk(wk, sl, k, src.to, t, p, first)
	return resume
}

// chainEvent applies the unit event the uniform u·total selects: a
// failure or a repair in one chain. It returns true for the failure of
// a unit replaced at once in a chain that needs all its units, which is
// a system failure that leaves no state behind.
func (w *Workspace) chainEvent(wk *worker, src csrc, u float64) bool {
	m := w.m
	// bound: the source's chains
	for c := src.chainStart; c < src.chainEnd; c++ {
		ch := m.chains[c]
		failRate := float64(ch.N-int(wk.failed[c])) * ch.Fail
		if u < failRate {
			if ch.Repair <= 0 {
				return ch.K == ch.N
			}
			wk.failed[c]++
			return false
		}
		u -= failRate
		repairRate := float64(wk.failed[c]) * ch.Repair
		if u < repairRate {
			wk.failed[c]--
			return false
		}
		u -= repairRate
	}
	return false
}
