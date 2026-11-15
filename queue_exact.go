package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

// QueueLongRun is a carried queue's long-run (stationary) wait and stay,
// where a closed form gives them: Erlang C for exponential work at any
// number of servers, Pollaczek–Khinchine for one server and any work.
// The wait's quantiles, at QuantileLevels(), lie in [WaitLo, WaitHi]:
// equal under Erlang C, and under Pollaczek–Khinchine from two
// discretisations of the law that bound it below and above. Reason says
// why there are none, Method is "" then.
type QueueLongRun struct {
	Method         string
	Wait, Stay     float64 // means, +Inf when infinite
	WaitLo, WaitHi [5]float64
	Reason         string
}

// pkCells is the grid of the Pollaczek–Khinchine quantiles: the
// interval of each is about one cell wide for every residual a wait
// sums.
const pkCells = 8192

// longRunQueues finds each queue's long-run figures.
func (m *Compiled) longRunQueues() []QueueLongRun {
	out := make([]QueueLongRun, len(m.queues))
	for c := range m.queues {
		out[c] = m.longRun(index(c))
	}
	return out
}

func (m *Compiled) longRun(c int32) QueueLongRun {
	q := m.queues[c]
	if q.reset {
		return QueueLongRun{Reason: "a reset queue empties each period, so it has no long-run line"}
	}
	if why := m.poissonInto(q.node); why != "" {
		return QueueLongRun{Reason: "the rounds reaching it are not a Poisson stream: " + why}
	}
	lam := 0.0
	for k := range m.srcs {
		lam += m.countMean[k] * m.reach[k][q.node]
	}
	work := m.pools[q.work]
	es := work.Mean() / q.speed
	rho := lam * es / float64(q.servers)
	switch {
	case !(rho < 1):
		return QueueLongRun{Reason: "its load " + ftoa(rho) + " is not below 1, so the line grows without bound"}
	case isExponential(work):
		return erlangC(lam, 1/es, int(q.servers))
	case q.servers == 1:
		return pollaczekKhinchine(lam, rho, q.speed, work)
	}
	return QueueLongRun{Reason: "no closed form for several servers unless the work is exponential"}
}

// isExponential reports a law whose log-survival is −x/E[X] at points
// from an eighth of its mean to 64 means: exponential, whatever family
// it is written in (a gamma or Weibull of shape 1 is one).
func isExponential(d dist.Distribution) bool {
	m := d.Mean()
	if lo, hi := d.Support(); lo != 0 || !math.IsInf(hi, 1) || !(m > 0) {
		return false
	}
	// bound: ten points
	for x := m / 8; x <= 64*m; x *= 2 {
		if want := -x / m; math.Abs(d.LogSurvival(x)-want) > 1e-12*math.Abs(want) {
			return false
		}
	}
	return true
}

// poissonInto says why the rounds reaching node v are not a Poisson
// stream at a constant rate, or "": each source reaching it must be
// plain Poisson at a constant rate, and the paths into it thin and delay
// them independently, with no queue and nothing drawn once a period.
func (m *Compiled) poissonInto(v int32) string {
	for k, s := range m.srcs {
		if m.countMean[k] > 0 && m.reach[k][v] > 0 && !plainPoisson(m, s) {
			return "source " + m.sourceIDs[k] + " is not Poisson at a constant rate"
		}
	}
	return m.pathsInto(v)
}

// plainPoisson reports a source of Poisson rounds at a constant rate.
func plainPoisson(m *Compiled, s csrc) bool {
	return isFixed(m.pools[s.pool]) && s.every <= 1 && s.repair == noIndex && s.chainEnd == s.chainStart &&
		s.factor == noIndex && s.schedule == noIndex && s.trigger == noIndex
}

// pathsInto says why the paths into node v do not keep a Poisson stream
// Poisson, or "": a queue on them, or a route drawn once a period.
func (m *Compiled) pathsInto(v int32) string {
	for u := range m.nodes {
		if index(u) == v || !m.reachedFrom([]int32{index(u)})[v] {
			continue
		}
		n := m.nodes[u]
		if n.queue != noIndex {
			return "it is reached through queue " + m.nodeIDs[u]
		}
		if n.thresh != noIndex && m.poolPeriod[m.fns[n.thresh].pool] != noIndex {
			return "node " + m.nodeIDs[u] + " routes on a value drawn once a period"
		}
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			if m.outs[j].probPool != noIndex {
				return "node " + m.nodeIDs[u] + " routes with a probability drawn once a period"
			}
		}
	}
	return ""
}

// erlangC is M/M/c with arrival rate lam and service rate mu a server:
// the wait exceeds t with probability C·e^(−(cμ−λ)t), C the probability
// of waiting.
func erlangC(lam, mu float64, c int) QueueLongRun {
	a := lam / mu
	rho := a / float64(c)
	term, sum := 1.0, 0.0
	// bound: the servers
	for k := range c {
		sum += term
		term *= a / float64(k+1)
	}
	tail := term / (1 - rho)
	pw := tail / (sum + tail)
	theta := float64(c)*mu - lam
	r := QueueLongRun{Method: "Erlang C", Wait: pw / theta}
	r.Stay = r.Wait + 1/mu
	for i, p := range QuantileLevels() {
		x := 0.0
		if pw > 1-p {
			x = math.Log(pw/(1-p)) / theta
		}
		r.WaitLo[i], r.WaitHi[i] = x, x
	}
	return r
}

// pollaczekKhinchine is M/G/1: the mean wait λE[S²]/(2(1−ρ)), and the
// wait's law, a geometric(ρ) number of the service's equilibrium
// residuals, whose density is P(S > x)/E[S].
func pollaczekKhinchine(lam, rho, speed float64, work dist.Distribution) QueueLongRun {
	es := work.Mean() / speed
	es2 := (work.Var() + work.Mean()*work.Mean()) / (speed * speed)
	r := QueueLongRun{Method: "Pollaczek–Khinchine", Wait: lam * es2 / (2 * (1 - rho))}
	r.Stay = r.Wait + es
	sbar := func(x float64) float64 { return work.Survival(x * speed) }
	beyond := func(x float64) float64 { return stopLoss(work, x*speed) / speed / es }
	x := pkExtent(rho, es, sbar, beyond)
	if math.IsInf(x, 1) {
		for i := range r.WaitLo {
			r.WaitLo[i], r.WaitHi[i] = math.NaN(), math.NaN()
		}
		return r
	}
	lo, hi := residualCells(es, x, pkCells, sbar, beyond)
	r.WaitLo = pkQuantiles(rho, lo, x/pkCells)
	r.WaitHi = pkQuantiles(rho, hi, x/pkCells)
	return r
}

// stopLoss is E[(X − x)⁺]: the mass above x times the mean excess there.
func stopLoss(d dist.Distribution, x float64) float64 {
	half, mass, err := dist.NewSplit(d, x, true)
	if err != nil || !(mass > 0) {
		return 0
	}
	return mass * math.Max(half.Mean()-x, 0)
}

// pkExtent is a grid length four times the lower bound's P99.9 on a
// coarse grid, quadrupling from 16 mean services over the idle share
// until it holds it; +Inf when 30 quadruplings do not. It sets only the
// grid: a quantile the fine grid cannot hold is +Inf there.
func pkExtent(rho, es float64, sbar, beyond func(float64) float64) float64 {
	const coarse = 512
	// bound: 30 quadruplings
	for x, i := 16*es/(1-rho), 0; i < 30; x, i = 4*x, i+1 {
		lo, _ := residualCells(es, x, coarse, sbar, beyond)
		if q := pkQuantiles(rho, lo, x/coarse); q[4] <= x/4 {
			return math.Max(4*q[4], x/coarse)
		}
	}
	return math.Inf(1)
}

const (
	// residualBlock is the cells whose total mass is taken exactly, from
	// the stop-loss at the block's ends.
	residualBlock = 64
	// residualSamples is the survival function's samples in a cell.
	residualSamples = 16
)

// residualCells discretises the equilibrium residual R, of density
// P(S > u)/E[S], on n cells of [0, x), n a multiple of residualBlock;
// beyond(y) is P(R > y). A block's mass is exact, from beyond at its
// ends, and as the survival falls its samples at a cell's sub-cells'
// right ends bound the cell's mass below and at their left ends above.
// hi puts each cell's lower bound on the cell's right end and the rest
// of the block's mass on the block's end, leaving out the mass beyond
// x, so it bounds R above; lo puts each cell's upper bound on its left
// end until the block's mass is reached, and the mass beyond x on x, so
// it bounds R below.
func residualCells(es, x float64, n int, sbar, beyond func(float64) float64) (lo, hi []float64) {
	lo, hi = make([]float64, n+1), make([]float64, n+1)
	h := x / float64(n)
	w := h / residualSamples / es
	left, prev := sbar(0), 1.0
	// bound: the grid's blocks
	for b := 0; b < n; b += residualBlock {
		next := beyond(float64(b+residualBlock) * h)
		mass := math.Max(prev-next, 0)
		prev = next
		sumLo, sumHi := 0.0, 0.0
		// bound: the cells of one block
		for j := b; j < b+residualBlock; j++ {
			a, u := 0.0, 0.0
			// bound: the samples of one cell
			for i := range residualSamples {
				right := sbar(float64(j)*h + float64(i+1)*h/residualSamples)
				a += right
				u += left
				left = right
			}
			hi[j+1] = math.Min(a*w, mass-sumHi)
			sumHi += hi[j+1]
			lo[j] = math.Min(u*w, mass-sumLo)
			sumLo += lo[j]
		}
		hi[b+residualBlock] += mass - sumHi
		lo[b] += mass - sumLo
	}
	lo[n] += prev
	return lo, hi
}

// pkQuantiles finds the quantiles at QuantileLevels() of a geometric(ρ)
// sum of the cells f, by Panjer's recursion, a quantile past the grid
// +Inf.
func pkQuantiles(rho float64, f []float64, h float64) [5]float64 {
	n := len(f)
	g := make([]float64, n)
	d := 1 - rho*f[0]
	g[0] = (1 - rho) / d
	// bound: the grid's cells
	for k := 1; k < n; k++ {
		s := 0.0
		// bound: the cells before k
		for i := 1; i <= k; i++ {
			s += f[i] * g[k-i]
		}
		g[k] = rho * s / d
	}
	var out [5]float64
	levels := QuantileLevels()
	cum, i := 0.0, 0
	// bound: the grid's cells
	for k := 0; k < n && i < len(levels); k++ {
		cum += g[k]
		// bound: the levels
		for i < len(levels) && cum >= levels[i] {
			out[i] = float64(k) * h
			i++
		}
	}
	// bound: the levels
	for ; i < len(levels); i++ {
		out[i] = math.Inf(1)
	}
	return out
}

// QueueLongRun returns queue c's long-run figures.
func (m *Compiled) QueueLongRun(c int) QueueLongRun { return m.queueLong[c] }
