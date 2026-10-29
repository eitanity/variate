package variate

import (
	"math"
	"math/rand/v2"
)

// An outage holding: an outcome whose holding time is the time the
// reliability block that fired the round stays down. The walk takes
// it from the block's own repairs, so a per-time cost over the outage
// follows the repair that caused it. For the analytics and the spill
// the outcome carries an outageBound in place of a holding
// distribution: its mean is the block's exact mean down time, its
// variance unknown, and its tail an exponential at twice the slowest
// repair, which sizes the window; an outage beyond the window fails the
// run with the window error, as any holding would.
type outageBound struct {
	mean float64 // u/f: the block's mean down time per failure
	tail float64 // mean of the exponential that bounds the tail
}

func (o outageBound) Fill(_ *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = o.mean
	}
}
func (o outageBound) Mean() float64                       { return o.mean }
func (outageBound) Var() float64                          { return math.NaN() }
func (outageBound) Support() (lo, hi float64)             { return 0, math.Inf(1) }
func (o outageBound) CDF(x float64) float64               { return 1 - o.Survival(x) }
func (outageBound) Laplace(complex128) (complex128, bool) { return 0, false }
func (outageBound) Nodes(_, _ []float64) bool             { return false }
func (o outageBound) Quantile(p float64) float64          { return -o.tail * math.Log1p(-p) }
func (o outageBound) QuantileUpper(q float64) float64     { return -o.tail * math.Log(q) }
func (o outageBound) Survival(x float64) float64 {
	if x <= 0 {
		return 1
	}
	return math.Exp(-x / o.tail)
}

// outageTailFactor is the multiple of the slowest repair's mean time
// the outage bound's tail takes.
const outageTailFactor = 2

// outageHolds gives every outage outcome its bound, once the reach is
// known: the node must be reached by exactly one source, a reliability
// block with a repair, whose figures give the mean down time.
func (m *Compiled) outageHolds() error {
	for j := range m.outs {
		if !m.outs[j].outage {
			continue
		}
		src, err := m.outageSource(j)
		if err != nil {
			return err
		}
		chains := m.chains[m.srcs[src].chainStart:m.srcs[src].chainEnd]
		f, u := ChainFigures(chains)
		slowest := math.Inf(1)
		for _, c := range chains {
			if c.Repair > 0 {
				slowest = math.Min(slowest, c.Repair)
			}
		}
		if !(f > 0) || math.IsInf(slowest, 1) {
			return modelError("outcome", m.outcomeNames[j], "an outage holding needs a block that goes down and is repaired")
		}
		m.outs[j].hold = m.holdPool(outageBound{mean: u / f, tail: outageTailFactor / slowest}, PerEvent)
	}
	return nil
}

// outageSource is the one source, a reliability block, that reaches
// the node of outage outcome j.
func (m *Compiled) outageSource(j int) (int, error) {
	v := m.outcomeNode[j]
	src := -1
	for k := range m.srcs {
		if m.reach[k][v] > 0 {
			if src >= 0 {
				return 0, modelError("outcome", m.outcomeNames[j], "an outage holding needs exactly one source reaching the node")
			}
			src = k
		}
	}
	if src < 0 || m.srcs[src].chainEnd == m.srcs[src].chainStart {
		return 0, modelError("outcome", m.outcomeNames[j], "an outage holding needs a reliability block as the source reaching the node")
	}
	return src, nil
}

// lookAhead runs a source's chains on from a system failure at t until
// the system is up again, returning that time, or limit when the outage
// outlasts it; the events it applies are the chains' real path, which
// the period's loop then resumes from.
func (w *Workspace) lookAhead(wk *worker, src csrc, t, limit float64) float64 {
	m := w.m
	drawn := 0
	// bound: the outage's unit events up to limit, a Poisson count, capped
	for !w.chainsUp(wk, src) {
		if drawn++; drawn > maxEventsPeriod {
			return limit
		}
		total := 0.0
		// bound: the source's chains
		for c := src.chainStart; c < src.chainEnd; c++ {
			ch := m.chains[c]
			total += float64(ch.N-int(wk.failed[c]))*ch.Fail + float64(wk.failed[c])*ch.Repair
		}
		if !(total > 0) {
			return limit
		}
		t += wk.ss.Control().ExpFloat64() / total
		if t >= limit {
			return limit
		}
		w.chainEvent(wk, src, wk.ss.Control().Float64()*total)
	}
	return t
}
