package variate

import (
	"math"

	"code.eitanity.com/variate/stats"
)

// Accum is the streamed statistics of one per-period quantity.
type Accum struct {
	Sum   stats.Neumaier // compensated sum over periods
	W     stats.Welford  // count, mean and M2 over periods
	Block stats.Welford  // over the means of full blocks
	Min   float64
	Max   float64
	Zero  int64          // periods with exactly 0
	Hist  *stats.LogHist // nil when the quantity has no histogram

	// With delays, coarse histograms of the full blocks in batchGroups
	// groups, block b in group b mod batchGroups, and the periods each
	// holds; nil without delays or a histogram. Where the groups are
	// independent timelines, GroupSum is each one's sum over its full
	// blocks, and GroupN is kept for every quantity.
	Groups   []*stats.LogHist
	GroupN   []int64
	GroupSum []stats.Neumaier
}

// N is the number of periods accumulated.
func (a *Accum) N() int64 { return a.W.N }

// Mean is the mean per period.
func (a *Accum) Mean() float64 { return a.W.Mean }

// StdDev is the sample standard deviation over periods.
func (a *Accum) StdDev() float64 { return a.W.StdDev() }

// StdErr is the standard error of the mean: from the periods without
// delays; with delays, from the means of the independent timelines where
// the run has them, else from the block means; NaN below 16 full blocks.
func (a *Accum) StdErr(delays bool) float64 {
	if !delays {
		return a.W.StdErr()
	}
	if a.Block.N < minBlocks {
		return math.NaN()
	}
	if a.GroupSum != nil {
		return groupErr(a.GroupN, func(g int) float64 { return a.GroupSum[g].Value() })
	}
	return a.Block.StdErr()
}

// groupErr is the standard error of the ratio Σx/Σn over independent
// groups, group g holding x(g) over n[g] periods, by the delta method:
// with G groups in use, the square root of G/(G−1) · Σ (n_g/N)² (x_g/n_g −
// x/N)². It is NaN with fewer than two groups in use.
func groupErr(n []int64, x func(g int) float64) float64 {
	var total int64
	var sum, used float64
	for g, ng := range n {
		if ng > 0 {
			total += ng
			sum += x(g)
			used++
		}
	}
	if used < 2 {
		return math.NaN()
	}
	mean := sum / float64(total)
	var ss float64
	for g, ng := range n {
		if ng > 0 {
			wgt := float64(ng) / float64(total)
			d := x(g)/float64(ng) - mean
			ss += wgt * wgt * d * d
		}
	}
	return math.Sqrt(ss * used / (used - 1))
}

// minBlocks is the number of full blocks the batch-means error needs.
const minBlocks = 16

const (
	// batchGroups is the number of groups the full blocks are dealt into
	// for the batch-means error of a proportion.
	batchGroups = 32
	// groupRes is the relative bin width of the group histograms.
	groupRes = 0.1
)

// Inflation is the ratio of the batch-means standard error of the share
// of periods at or above x to its binomial standard error: 1 for
// independent periods, and with delays measured from the group
// histograms at the coarse bin holding x. NaN below minBlocks full
// blocks. Where every or no full-block period reaches x, the groups
// cannot measure it, and it is 1.
func (a *Accum) Inflation(x float64) float64 {
	if a.Groups == nil {
		return 1
	}
	if a.Block.N < minBlocks {
		return math.NaN()
	}
	var n, c int64
	for g, h := range a.Groups {
		n += a.GroupN[g]
		c += h.AtOrAbove(x)
	}
	if c == 0 || c == n {
		return 1
	}
	p := float64(c) / float64(n)
	batch := groupErr(a.GroupN, func(g int) float64 { return float64(a.Groups[g].AtOrAbove(x)) })
	return batch / math.Sqrt(p*(1-p)/float64(n))
}

func (a *Accum) reset() {
	a.Sum = stats.Neumaier{}
	a.W = stats.Welford{}
	a.Block = stats.Welford{}
	a.Min, a.Max = math.Inf(1), math.Inf(-1)
	a.Zero = 0
	if a.Hist != nil {
		a.Hist.Reset()
	}
	for _, h := range a.Groups {
		h.Reset()
	}
	clear(a.GroupN)
	clear(a.GroupSum)
}

// Result is the outcome of a run.
type Result struct {
	Seed      uint64
	Periods   int64
	Converged bool
	Stopped   bool
	Binding   string
	Unmet     []string

	Nodes, Outcomes, Categories, Sources []string

	Events       []uint64   // [source]
	Reach        [][]uint64 // [source][node]
	OutcomeCount [][]uint64 // [source][outcome]

	Total      Accum
	ByCategory []Accum
	BySource   []Accum
	EntriesPer []Accum // [node]
	Contrib    []Accum // [node]; zero for nodes without functions
	Layers     []Accum // [layer]: the layer's payout per period
	Triggers   []Accum // [trigger]: the sum the trigger tests per period
	QueueCols  []Accum // [queue×queueCols]: each queue's joined, served, wait, stay, length, busy and unfinished per period
	Levels     []Accum // [Config.At]: 1 in a period reaching the value, else 0
	Queues     []QueueResult

	Steps         uint64
	BeyondHorizon float64
	TailPeriods   int64
	// Per tail threshold (Compiled.TailThresholds), the periods whose
	// total reached it and their total, categories and sources summed,
	// in the row order, for the tail shares.
	TailCounts []int64
	TailSums   [][]float64

	Series *Series // raw per-period rows; fixed runs with Config.Series only

	// ByPeriod is a started run's statistics per period of its horizon,
	// across the repeats; nil in steady state. The fields above then
	// describe the horizon: one row per repeat, its periods summed.
	ByPeriod *PeriodStats
}

// QueueResult is a queue's statistics over the rounds it served: their
// waits and their stays, waiting and service.
type QueueResult struct {
	wait, stay itemAccum
}

// Wait summarises the waits of the rounds served.
func (q *QueueResult) Wait() ItemStat { return q.wait.stat() }

// Stay summarises the stays, wait and service, of the rounds served.
func (q *QueueResult) Stay() ItemStat { return q.stay.stat() }

// PeriodStats holds, per period of a started run's horizon and per
// quantity, the mean and spread across the repeats.
type PeriodStats struct {
	l layout
	h int
	w []stats.Welford // [period×width]
	// Per period and queue, the co-moment of the waits' and the stays'
	// sums with the rounds served, for the ratio's error.
	waitCo, stayCo []float64 // [period×queues]
}

// PeriodStat is one quantity in one period of the horizon, across the
// repeats: their number, the mean, its standard error and the standard
// deviation.
type PeriodStat struct {
	N                    int64
	Mean, StdErr, StdDev float64
}

// Periods is the horizon.
func (s *PeriodStats) Periods() int { return s.h }

func (s *PeriodStats) at(k int, i int) PeriodStat {
	w := s.w[k*s.l.width+i]
	return PeriodStat{N: w.N, Mean: w.Mean, StdErr: w.StdErr(), StdDev: w.StdDev()}
}

// Total is the period total in period k of the horizon, from 0.
func (s *PeriodStats) Total(k int) PeriodStat { return s.at(k, s.l.total()) }

// Category is category c's value in period k.
func (s *PeriodStats) Category(c, k int) PeriodStat { return s.at(k, s.l.cat(index(c))) }

// Source is source k's value in period p.
func (s *PeriodStats) Source(k, p int) PeriodStat { return s.at(p, s.l.src(index(k))) }

// Entries is node v's entries in period k.
func (s *PeriodStats) Entries(v, k int) PeriodStat { return s.at(k, s.l.entry(index(v))) }

// Contrib is node v's booked value in period k.
func (s *PeriodStats) Contrib(v, k int) PeriodStat { return s.at(k, s.l.contrib(index(v))) }

// Trigger is the sum trigger i tested in period k.
func (s *PeriodStats) Trigger(i, k int) PeriodStat { return s.at(k, s.l.trig(i)) }

// add folds one repeat's rows, a row per period of the horizon: each
// queue's co-moments first, from the means before this repeat.
func (s *PeriodStats) add(rows []float64) {
	width := s.l.width
	for k := range s.h {
		for c := range s.l.queues {
			at := k*s.l.queues + c
			served := k*width + s.l.queue(c, qServed)
			n := float64(s.w[served].N + 1)
			dy := rows[served] - s.w[served].Mean
			for _, x := range [2]struct {
				co  []float64
				col int
			}{{s.waitCo, qWait}, {s.stayCo, qStay}} {
				i := k*width + s.l.queue(c, x.col)
				x.co[at] += (n - 1) / n * (rows[i] - s.w[i].Mean) * dy
			}
		}
	}
	for i := range s.w {
		s.w[i].Add(rows[i])
	}
}

// QueueRatio is a queue's mean wait or stay of the rounds that reached it
// in a period and were served, with its standard error across the
// repeats by the delta method; NaN where none was served.
type QueueRatio struct{ Mean, StdErr float64 }

// Queue is queue c's column col (Joined … Unfinished) in period k.
func (s *PeriodStats) Queue(c, col, k int) PeriodStat { return s.at(k, s.l.queue(c, col)) }

// Wait is queue c's mean wait in period k; Stay its mean stay.
func (s *PeriodStats) Wait(c, k int) QueueRatio { return s.ratio(c, k, qWait, s.waitCo) }

// Stay is queue c's mean stay of the rounds that reached it in period k.
func (s *PeriodStats) Stay(c, k int) QueueRatio { return s.ratio(c, k, qStay, s.stayCo) }

func (s *PeriodStats) ratio(c, k, col int, co []float64) QueueRatio {
	x, y := s.w[k*s.l.width+s.l.queue(c, col)], s.w[k*s.l.width+s.l.queue(c, qServed)]
	if !(y.Mean > 0) || y.N < 2 {
		return QueueRatio{Mean: math.NaN(), StdErr: math.NaN()}
	}
	r := x.Mean / y.Mean
	cov := co[k*s.l.queues+c] / float64(y.N-1)
	v := (x.Var() - 2*r*cov + r*r*y.Var()) / (float64(y.N) * y.Mean * y.Mean)
	return QueueRatio{Mean: r, StdErr: math.Sqrt(math.Max(v, 0))}
}

// Queues are the queue column names, in the order of Queue's col.
func QueueColumns() []string {
	return []string{"joined", "served", "wait_sum", "stay_sum", "line_at_end", "busy", "unfinished"}
}

// Series holds every closed period's row of quantities: total, categories,
// sources, entries per node, contribution per node, layers and the sum
// each trigger tests.
type Series struct {
	l       layout
	rows    []float64 // [period×width]
	n       int64
	horizon int64 // a started run's periods per repeat, which follow each other in the rows; 0 in steady state
}

// Periods is the number of periods recorded: in a started run, the
// repeats times the horizon.
func (s *Series) Periods() int64 { return s.n }

// Horizon is a started run's periods per repeat, row p being period
// p mod Horizon of repeat p / Horizon; 0 in steady state.
func (s *Series) Horizon() int64 { return s.horizon }

func (s *Series) at(p int64, i int) float64 { return s.rows[int(p)*s.l.width+i] }

// Total is the period total of period p.
func (s *Series) Total(p int64) float64 { return s.at(p, s.l.total()) }

// Category is category c's value in period p.
func (s *Series) Category(c int, p int64) float64 { return s.at(p, s.l.cat(index(c))) }

// Source is source k's value in period p.
func (s *Series) Source(k int, p int64) float64 { return s.at(p, s.l.src(index(k))) }

// Entries is the number of entries at node v in period p.
func (s *Series) Entries(v int, p int64) float64 { return s.at(p, s.l.entry(index(v))) }

// Contrib is node v's booked value in period p.
func (s *Series) Contrib(v int, p int64) float64 { return s.at(p, s.l.contrib(index(v))) }

// Column copies one quantity's per-period values into dst, which must
// have Periods() entries, and returns it.
func (s *Series) column(i int, dst []float64) []float64 {
	for p := range dst {
		dst[p] = s.rows[p*s.l.width+i]
	}
	return dst
}

// TotalColumn copies the period totals into dst.
func (s *Series) TotalColumn(dst []float64) []float64 { return s.column(s.l.total(), dst) }

// LayerColumn copies a layer's payouts into dst.
func (s *Series) LayerColumn(j int, dst []float64) []float64 { return s.column(s.l.layer(j), dst) }

// TriggerColumn copies the sums trigger i tested into dst.
func (s *Series) TriggerColumn(i int, dst []float64) []float64 { return s.column(s.l.trig(i), dst) }

// EntriesColumn copies node v's entries into dst.
func (s *Series) EntriesColumn(v int, dst []float64) []float64 {
	return s.column(s.l.entry(index(v)), dst)
}

// CategoryColumn copies category c's values into dst.
func (s *Series) CategoryColumn(c int, dst []float64) []float64 {
	return s.column(s.l.cat(index(c)), dst)
}

// SourceColumn copies source k's values into dst.
func (s *Series) SourceColumn(k int, dst []float64) []float64 {
	return s.column(s.l.src(index(k)), dst)
}
