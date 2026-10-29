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
}

// N is the number of periods accumulated.
func (a *Accum) N() int64 { return a.W.N }

// Mean is the mean per period.
func (a *Accum) Mean() float64 { return a.W.Mean }

// StdDev is the sample standard deviation over periods.
func (a *Accum) StdDev() float64 { return a.W.StdDev() }

// StdErr is the standard error of the mean: from the periods without
// delays, from the block means with delays; NaN below 16 full blocks.
func (a *Accum) StdErr(delays bool) float64 {
	if !delays {
		return a.W.StdErr()
	}
	if a.Block.N < minBlocks {
		return math.NaN()
	}
	return a.Block.StdErr()
}

// minBlocks is the number of full blocks the batch-means error needs.
const minBlocks = 16

func (a *Accum) reset() {
	a.Sum = stats.Neumaier{}
	a.W = stats.Welford{}
	a.Block = stats.Welford{}
	a.Min, a.Max = math.Inf(1), math.Inf(-1)
	a.Zero = 0
	if a.Hist != nil {
		a.Hist.Reset()
	}
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

	Steps         uint64
	BeyondHorizon float64
	TailPeriods   int64
	// Per tail threshold (Compiled.TailThresholds), the periods whose
	// total reached it and their total, categories and sources summed,
	// in the row order, for the tail shares.
	TailCounts []int64
	TailSums   [][]float64

	Series *Series // raw per-period rows; fixed runs with Config.Series only
}

// Series holds every closed period's row of quantities: total, categories,
// sources, entries per node and contribution per node.
type Series struct {
	l    layout
	rows []float64 // [period×width]
	n    int64
}

// Periods is the number of periods recorded.
func (s *Series) Periods() int64 { return s.n }

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

// EntriesColumn copies node v's entries into dst.
func (s *Series) EntriesColumn(v int, dst []float64) []float64 {
	return s.column(s.l.entry(index(v)), dst)
}

// CategoryColumn copies category c's values into dst.
func (s *Series) CategoryColumn(c int, dst []float64) []float64 {
	return s.column(s.l.cat(index(c)), dst)
}
