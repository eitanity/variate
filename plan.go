package variate

import (
	"errors"
	"math"
	"sort"
)

const (
	defaultMinHits    = 1000
	defaultContribRSE = 0.05
	defaultTargetRSE  = 0.01
	defaultTailP      = 0.999
	defaultTailHits   = 100
	defaultMaxPeriods = 1e11
	defaultMaxSteps   = 1e13

	critMinHits = "min-hits"
	critContrib = "contrib-rse"
	critTarget  = "target-rse"
	critTail    = "tail"
)

// criteria holds the convergence settings with their defaults applied.
type criteria struct {
	minHits    uint64
	contribRSE float64
	targetRSE  float64
	tailP      float64
	tailHits   uint64
	maxPeriods int64
	maxSteps   uint64
}

func criteriaFrom(cfg Config) criteria {
	c := criteria{
		minHits: cfg.MinHits, contribRSE: cfg.ContribRSE, targetRSE: cfg.TargetRSE,
		tailP: cfg.TailP, tailHits: cfg.TailHits, maxPeriods: cfg.MaxPeriods, maxSteps: cfg.MaxSteps,
	}
	if c.minHits == 0 {
		c.minHits = defaultMinHits
	}
	if c.contribRSE <= 0 {
		c.contribRSE = defaultContribRSE
	}
	if c.targetRSE <= 0 {
		c.targetRSE = defaultTargetRSE
	}
	if c.tailP <= 0 || c.tailP >= 1 {
		c.tailP = defaultTailP
	}
	if c.tailHits == 0 {
		c.tailHits = defaultTailHits
	}
	if c.maxPeriods <= 0 {
		c.maxPeriods = defaultMaxPeriods
	}
	if c.maxSteps == 0 {
		c.maxSteps = defaultMaxSteps
	}
	return c
}

// planRow is the plan's internal row with the quantity it watches.
type planRow struct {
	PlanRow
	kind  string
	index int // outcome, node, or row quantity the criterion watches
}

// buildPlan sizes every criterion from the analytic values. Rows whose
// variance is unknown (delays, pair budget) get PeriodsNeeded 0 and are
// governed by the observed check alone.
func (m *Compiled) buildPlan(e *Expectation, c criteria) []planRow {
	var rows []planRow
	for j, h := range e.OutcomeRate {
		if h <= 0 {
			continue
		}
		rows = append(rows, planRow{kind: critMinHits, index: j, PlanRow: PlanRow{
			Criterion: critMinHits, Driver: m.outcomeNames[j], PlannedPerPeriod: h,
			PeriodsNeeded: periodsFor(float64(c.minHits) / h),
		}})
	}
	for v := range m.nodes {
		if m.nodes[v].fnStart == m.nodes[v].rateEnd || e.Contrib[v] <= 0 {
			continue
		}
		need := e.ContribVar[v] / (c.contribRSE * c.contribRSE * e.Contrib[v] * e.Contrib[v])
		rows = append(rows, planRow{kind: critContrib, index: v, PlanRow: PlanRow{
			Criterion: critContrib, Driver: m.nodeIDs[v], PlannedPerPeriod: e.EntriesPer[v], PeriodsNeeded: periodsFor(need),
		}})
	}
	l := newLayout(m)
	target := func(name string, q int, mean, variance float64) {
		if mean <= 0 {
			return
		}
		rows = append(rows, planRow{kind: critTarget, index: q, PlanRow: PlanRow{
			Criterion: critTarget, Driver: name, PlannedPerPeriod: mean,
			PeriodsNeeded: periodsFor(variance / (c.targetRSE * c.targetRSE * mean * mean)),
		}})
	}
	target("total", l.total(), e.Mean, e.Var)
	for i, name := range m.categories {
		target("category "+name, l.cat(index(i)), e.MeanByCategory[i], e.VarByCategory[i])
	}
	for k, name := range m.sourceIDs {
		target("source "+name, l.src(index(k)), e.MeanBySource[k], e.VarBySource[k])
	}
	rows = append(rows, planRow{kind: critTail, PlanRow: PlanRow{
		Criterion: critTail, Driver: totalSubject, PlannedPerPeriod: 1 - c.tailP,
		PeriodsNeeded: periodsFor(float64(c.tailHits) / (1 - c.tailP)),
	}})
	return rows
}

// periodsFor rounds a period count up; an unknown (NaN or infinite) need
// gives 0, meaning "observed check only".
func periodsFor(x float64) int64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return 0
	}
	return int64(math.Ceil(x))
}

// Plan returns the sizing table: one row per criterion, with the periods
// it needs; Observed and Met are unset.
func (m *Compiled) Plan() []PlanRow {
	out := make([]PlanRow, len(m.plan))
	for i, r := range m.plan {
		out[i] = r.PlanRow
	}
	return out
}

// PlanPeriods is the plan's run length: the largest need, rounded up to
// whole batches of blocks.
func (m *Compiled) PlanPeriods(batchPeriods int64) int64 {
	need := int64(0)
	for _, r := range m.plan {
		need = max(need, r.PeriodsNeeded)
	}
	if batchPeriods <= 0 {
		return need
	}
	return (need + batchPeriods - 1) / batchPeriods * batchPeriods
}

// Binding returns the criterion and driver with the largest need.
func (m *Compiled) Binding() (criterion, driver string) {
	best := -1
	for i, r := range m.plan {
		if best < 0 || r.PeriodsNeeded > m.plan[best].PeriodsNeeded {
			best = i
		}
	}
	if best < 0 {
		return "", ""
	}
	return m.plan[best].Criterion, m.plan[best].Driver
}

// checkCaps refuses a plan that exceeds MaxPeriods or MaxSteps, listing
// the rows that cannot be met within the budget, unless AllowUnconverged.
func (m *Compiled) checkCaps(c criteria, steps float64, allow bool) error {
	var over []string
	for _, r := range m.plan {
		if r.PeriodsNeeded > c.maxPeriods || float64(r.PeriodsNeeded)*steps > float64(c.maxSteps) {
			over = append(over, r.Criterion+" "+r.Driver+" needs "+itoa(int(r.PeriodsNeeded))+" periods")
		}
	}
	if len(over) == 0 || allow {
		return nil
	}
	sort.Strings(over)
	msg := "plan exceeds the caps: "
	for i, o := range over {
		if i > 0 {
			msg += "; "
		}
		msg += o
	}
	return errors.New(msg + "; raise MaxPeriods/MaxSteps or set AllowUnconverged")
}
