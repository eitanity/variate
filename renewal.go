package variate

import (
	"errors"
	"math"
	"math/cmplx"
)

// A source's count per period follows one of three laws: mixed Poisson
// at its rate; every n-th arrival of that Poisson stream, whose
// inter-arrival times are Erlang; or the rate while up with a silent
// repair after each firing, an alternating renewal. The last two keep
// state within a source's own stream only, so the walk is unchanged
// elsewhere, and every block starts the state in equilibrium.

const (
	maxChainUnits    = 64
	maxEvery         = 1 << 20
	maxResidualDraws = 1 << 16
	residualTail     = 1e-12
	poissonSpread    = 12 // standard deviations of Poisson mass kept, plus 60
	poissonSlack     = 60
	maxErlangWork    = 1 << 28 // rate quadrature nodes × mass function length, at compile
	maxEventsPeriod  = 1 << 24 // arrivals one source may draw in one period before the run fails
)

// countLaws fills, per source, the mean and second factorial moment of
// its count per period and, for the Erlang law, the count's mass
// function mixed over the rate. The repair law has an exact mean and
// no closed second moment.
func (m *Compiled) countLaws() error {
	n := len(m.srcs)
	m.countMean = make([]float64, n)
	m.countN2 = make([]float64, n)
	m.countPMF = make([][]float64, n)
	m.countTail = make([]float64, n)
	for k, s := range m.srcs {
		d := m.pools[s.pool]
		switch {
		case s.chainEnd > s.chainStart:
			m.countMean[k], _ = ChainFigures(m.chains[s.chainStart:s.chainEnd])
			m.countN2[k] = math.NaN()
		case s.repair != noIndex:
			lam, h := d.Mean(), m.pools[s.repair].Mean()
			if lam > 0 {
				m.countMean[k] = 1 / (1/lam + h)
			}
			m.countN2[k] = math.NaN()
		case s.every > 1:
			m.countMean[k] = d.Mean() / float64(s.every)
			pmf, tail, err := m.erlangMixed(d, int(s.every))
			if err != nil {
				return modelError("source", m.sourceIDs[k], err.Error())
			}
			m.countPMF[k], m.countTail[k] = pmf, tail
			m.countN2[k] = factorial2(m.countPMF[k])
		default:
			m.countMean[k] = d.Mean()
			m.countN2[k] = d.Var() + m.countMean[k]*m.countMean[k]
			for _, pool := range [2]int32{s.factor, s.schedule} {
				if pool != noIndex {
					f := m.pools[pool]
					m.countMean[k] *= f.Mean()
					m.countN2[k] *= f.Var() + f.Mean()*f.Mean()
				}
			}
		}
	}
	return nil
}

// countPair is E[N_k N_l] for two different sources: the product of
// their means, times E[M²]/E[M]² for each multiplier they share, a
// factor or a schedule.
func (m *Compiled) countPair(k, l int) float64 {
	pair := m.countMean[k] * m.countMean[l]
	shared := [2][2]int32{{m.srcs[k].factor, m.srcs[l].factor}, {m.srcs[k].schedule, m.srcs[l].schedule}}
	for _, pools := range shared {
		if f := pools[0]; f != noIndex && f == pools[1] {
			mean := m.poolMean[f]
			pair *= (m.pools[f].Var() + mean*mean) / (mean * mean)
		}
	}
	return pair
}

// factorAt is the product of source k's multipliers, its factor and
// its schedule, at the drawn period values; 1 without.
func (m *Compiled) factorAt(k int, drawn []float64) float64 {
	f := 1.0
	if p := m.srcs[k].factor; p != noIndex {
		f *= drawn[p]
	}
	if p := m.srcs[k].schedule; p != noIndex {
		f *= drawn[p]
	}
	return f
}

// schedule is a compiled schedule: its pool and its cycle of values.
type schedule struct {
	pool   int32
	values []float64
}

const maxScheduleLength = 1 << 16

// at is the schedule's multiplier for period p, the burn-in's negative
// periods wrapping from the cycle's end.
func (sc schedule) at(p int64) float64 {
	n := int64(len(sc.values))
	return sc.values[((p%n)+n)%n]
}

// rateGiven and rate2Given are E[λ_k] and E[λ_k²] of a Poisson source
// at the drawn period values: its own rate's moments scaled by the
// factor drawn.
func (m *Compiled) rateGiven(k int, drawn []float64) float64 {
	return m.poolMean[m.srcs[k].pool] * m.factorAt(k, drawn)
}

func (m *Compiled) rate2Given(k int, drawn []float64) float64 {
	d := m.pools[m.srcs[k].pool]
	f := m.factorAt(k, drawn)
	return (d.Var() + d.Mean()*d.Mean()) * f * f
}

// erlangMixed is the mass function of every n-th arrival's count per
// period, averaged over the rate's distribution by its quadrature rule,
// with the Poisson mass the truncation left out.
func (m *Compiled) erlangMixed(d distribution, n int) ([]float64, float64, error) {
	lo, hi := d.Support()
	if lo == hi {
		if kmax := erlangLength(hi); kmax > maxErlangWork {
			return nil, 0, errors.New("every: a rate of " + ftoa(hi) + " per period makes the count's mass function " + itoa(kmax) + " long, above " + itoa(maxErlangWork))
		}
		p, t := erlangPMF(lo, n)
		return p, t, nil
	}
	rule := rateRule(d)
	if work := len(rule.x) * erlangLength(hi); work > maxErlangWork {
		return nil, 0, errors.New("every: the uncertain rate needs " + itoa(len(rule.x)) + " quadrature nodes over a mass function " + itoa(erlangLength(hi)) + " long, above " + itoa(maxErlangWork) + "; a constant rate or a narrower one")
	}
	var pmf []float64
	tail := 0.0
	for i, lam := range rule.x {
		p, t := erlangPMF(lam, n)
		if len(p) > len(pmf) {
			pmf = append(pmf, make([]float64, len(p)-len(pmf))...)
		}
		for j, x := range p {
			pmf[j] += rule.w[i] * x
		}
		tail += rule.w[i] * t
	}
	return pmf, tail, nil
}

// erlangLength is the Poisson mass kept for a rate: its mean, twelve
// standard deviations and a margin.
func erlangLength(lam float64) int {
	if !(lam > 0) || math.IsInf(lam, 1) {
		return 1
	}
	return int(lam + poissonSpread*math.Sqrt(lam) + poissonSlack)
}

// erlangPMF is the mass function of the count of every n-th arrival of
// a Poisson stream of rate lam over one period, in equilibrium: the
// stream's count K folded by the phase, uniform over the n positions,
// which puts a triangular kernel of width n about each multiple of n.
// The Poisson mass beyond the kept range is returned as the tail.
func erlangPMF(lam float64, n int) ([]float64, float64) {
	if !(lam > 0) {
		return []float64{1}, 0
	}
	kmax := erlangLength(lam)
	pk := make([]float64, kmax+1)
	total := 0.0
	logLam := math.Log(lam)
	// bound: kmax, the kept range of the Poisson mass
	for j := 0; j <= kmax; j++ {
		lg, _ := math.Lgamma(float64(j) + 1)
		pk[j] = math.Exp(-lam + float64(j)*logLam - lg)
		total += pk[j]
	}
	pmf := make([]float64, kmax/n+1)
	for i := range pmf {
		centre := n * i
		// bound: the kernel's 2n − 1 positions
		for j := max(0, centre-n+1); j <= min(kmax, centre+n-1); j++ {
			pmf[i] += float64(n-abs(j-centre)) / float64(n) * pk[j]
		}
	}
	return pmf, math.Max(0, 1-total)
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// factorial2 is E[N(N − 1)] under a mass function.
func factorial2(pmf []float64) float64 {
	s := 0.0
	for i, p := range pmf {
		s += float64(i*(i-1)) * p
	}
	return s
}

// arrivalsOf is the work a source's count law draws per period beyond
// its rounds: the stream's arrivals for every n-th and repair, and the
// units' failures and repairs, twice the failures in equilibrium, for
// chains; 0 for a Poisson source, whose rounds are its arrivals.
func (m *Compiled) arrivalsOf(k int) float64 {
	s := m.srcs[k]
	switch {
	case s.chainEnd > s.chainStart:
		work := 0.0
		for _, c := range m.chains[s.chainStart:s.chainEnd] {
			work += 2 * float64(c.N) * c.Fail * (1 - c.unitDown())
		}
		return work
	case s.every > 1 || s.repair != noIndex:
		return m.poolMean[s.pool]
	}
	return 0
}

// countGF is the probability generating function of source k's count
// at z with its rate scaled by f, for the laws with a closed form; ok
// is false for the repair law and for a mixed Poisson, which the caller
// integrates itself.
func (m *Compiled) countGF(k int, z complex128, f float64) (complex128, bool) {
	s := m.srcs[k]
	if s.repair != noIndex || s.chainEnd > s.chainStart {
		return cmplx.NaN(), false
	}
	if pmf := m.countPMF[k]; pmf != nil {
		return polyAt(pmf, z), true
	}
	return m.pools[s.pool].Laplace(complex(f, 0) * (1 - z))
}

// polyAt evaluates Σ p_i z^i by Horner's rule.
func polyAt(p []float64, z complex128) complex128 {
	acc := complex(0, 0)
	for i := len(p) - 1; i >= 0; i-- {
		acc = acc*z + complex(p[i], 0)
	}
	return acc
}

// startSources puts every renewal source in its equilibrium state at a
// block's start, from the control stream: the Erlang phase uniform over
// its positions; the repair law down with the stationary probability,
// for the equilibrium residual of its repair time.
func (w *Workspace) startSources(wk *worker, start float64) {
	m := w.m
	for k, s := range m.srcs {
		switch {
		case s.chainEnd > s.chainStart:
			w.startChains(wk, s)
			wk.chainT[k] = start
		case s.repair != noIndex:
			wk.downUntil[k] = math.Inf(-1)
			lam, h := m.poolMean[s.pool], m.poolMean[s.repair]
			if lam > 0 && h > 0 && wk.ss.Control().Float64() < h/(1/lam+h) {
				wk.downUntil[k] = start + w.residual(wk, s.repair)
			}
		case s.every > 1:
			wk.phase[k] = int32(wk.ss.Control().Float64() * float64(s.every))
		}
	}
}

// residual draws the equilibrium residual of a repair time: a uniform
// share of a length-biased draw, which rejection takes from the pool
// against the support's end, or the far quantile of an unbounded one.
// A constant needs no rejection.
func (w *Workspace) residual(wk *worker, pool int32) float64 {
	d := w.m.pools[pool]
	lo, hi := d.Support()
	if lo == hi {
		return wk.ss.Control().Float64() * hi
	}
	if math.IsInf(hi, 1) {
		hi = d.Quantile(1 - residualTail)
	}
	x := 0.0
	// bound: maxResidualDraws, after which the last draw stands
	for i := 0; i < maxResidualDraws; i++ {
		x = wk.ss.Sampler(int(pool)).Next()
		if wk.ss.Control().Float64()*hi < x {
			break
		}
	}
	return wk.ss.Control().Float64() * x
}

// runRenewalSource emits the period's rounds of a renewal source: the
// underlying arrivals come as exponential gaps at the period's rate,
// and each either advances the Erlang phase, firing when it completes,
// or fires when the source is up and silences it for a repair.
func (w *Workspace) runRenewalSource(wk *worker, sl *slot, k int32, p, first int64) {
	m := w.m
	src := m.srcs[k]
	lam := wk.period[m.poolPeriod[src.pool]]
	if !(lam > 0) {
		return
	}
	end := float64(p) + 1
	drawn := 0
	// bound: the period's underlying arrivals, a Poisson count, capped
	for t := float64(p) + wk.ss.Control().ExpFloat64()/lam; t < end && sl.err == nil; t += wk.ss.Control().ExpFloat64() / lam {
		if drawn++; drawn > maxEventsPeriod {
			sl.err = eventsCapError(m.sourceIDs[k], p)
			return
		}
		if src.repair != noIndex {
			if t < wk.downUntil[k] {
				continue
			}
			wk.downUntil[k] = t + w.draw(wk, src.repair)
		} else {
			wk.phase[k]++
			if wk.phase[k] < src.every {
				continue
			}
			wk.phase[k] = 0
		}
		if p >= 0 {
			sl.events[k]++
		}
		if w.stream {
			w.beginRound(wk, t)
		}
		w.walk(wk, sl, k, src.to, t, p, first)
	}
}
