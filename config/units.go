package config

import (
	"errors"
	"math"
	"strconv"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// The conversions a model built as data needs as much as one read from
// a file: a time unit into the model's period, and a reliability block
// into the chains or the rate a source runs on. The file reader builds
// through these, so a model built either way is the same.

// HoursIn is the hours one unit holds: "hours", "days" or "weeks"; 0
// for any other.
func HoursIn(unit string) float64 {
	switch unit {
	case "hours":
		return 1
	case "days":
		return 24
	case "weeks":
		return 168
	}
	return 0
}

// UnitScale is the factor a value given in unit takes into a model whose
// period is periodHours hours: for a time (a holding time, a repair
// time), hours ÷ periodHours; for a rate (a frequency, a per-time value),
// periodHours ÷ hours. A unit of "" or "periods" is 1.
func UnitScale(unit string, periodHours float64, time bool) (float64, error) {
	if unit == "" || unit == unitPeriods {
		return 1, nil
	}
	hours := HoursIn(unit)
	switch {
	case hours == 0:
		return 0, errors.New("unit " + unit + ": hours, days, weeks or periods")
	case !(periodHours > 0) || math.IsInf(periodHours, 1):
		return 0, errors.New("unit: " + unit + " used but period: not declared")
	case time:
		return hours / periodHours, nil
	}
	return periodHours / hours, nil
}

// BlockChain is a reliability component, or n identical ones of which k
// must work, as the chain a source runs on: mtbf the mean up time and
// mttr the mean repair time, 0 for a unit replaced at once, both in
// unit (hours, days or weeks), with the ranges a sweep moves them over
// in that unit too, or nil. The rates are per period of periodHours
// hours. A series of blocks is the chains of its parts, in order.
func BlockChain(k, n int, mtbf, mttr float64, mtbfRange, mttrRange *variate.InputRange, unit string, periodHours float64) (variate.Chain, error) {
	hours := HoursIn(unit)
	switch {
	case hours == 0:
		return variate.Chain{}, errors.New("a block needs unit: hours, days or weeks")
	case !(periodHours > 0) || math.IsInf(periodHours, 1):
		return variate.Chain{}, errors.New("a block needs period: declared")
	case n < 1 || n > maxUnits || k < 1 || k > n:
		return variate.Chain{}, errors.New("a block needs 1 ≤ k ≤ n ≤ " + strconv.Itoa(maxUnits))
	case !(mtbf > 0) || math.IsInf(mtbf, 1):
		return variate.Chain{}, errors.New("mtbf must be positive")
	case mttr < 0 || math.IsInf(mttr, 1) || math.IsNaN(mttr):
		return variate.Chain{}, errors.New("mttr must be zero or positive")
	case mttr == 0 && mttrRange != nil:
		return variate.Chain{}, errors.New("an mttr of 0 is a unit replaced at once, with no repair time for a range to move")
	}
	toPeriods := hours / periodHours
	ch := variate.Chain{K: k, N: n, Fail: periodHours / (mtbf * hours), MTBFRange: inPeriods(mtbfRange, toPeriods)}
	if mttr > 0 {
		ch.Repair = periodHours / (mttr * hours)
		ch.MTTRRange = inPeriods(mttrRange, toPeriods)
	}
	return ch, nil
}

// inPeriods is a range given in a block's unit, in periods; a range of
// factors is kept as it is.
func inPeriods(r *variate.InputRange, toPeriods float64) *variate.InputRange {
	if r == nil || r.Factors {
		return r
	}
	return &variate.InputRange{Low: r.Low * toPeriods, High: r.High * toPeriods}
}

// BlockSource sets a reliability diagram's chains on source s and gives
// the block's figures. With a repair time on any unit the source runs the
// chains, simulated exactly; with every unit replaced at once the diagram
// is a Poisson source at its stationary rate, which has an exact
// variance, and then no unit's time may carry a range.
func BlockSource(id string, chains []variate.Chain, periodHours float64, s *variate.Source) (Block, error) {
	if !(periodHours > 0) || math.IsInf(periodHours, 1) {
		return Block{}, errors.New("a block needs period: declared")
	}
	if len(chains) == 0 {
		return Block{}, errors.New("a block needs at least one component")
	}
	rate, u := variate.ChainFigures(chains)
	b := Block{Source: id, Rate: rate, MTBF: periodHours / rate, Unavailability: u}
	repaired := false
	for _, ch := range chains {
		repaired = repaired || ch.Repair > 0
	}
	if repaired {
		s.Chains = chains
		return b, nil
	}
	for _, ch := range chains {
		if ch.MTBFRange != nil {
			return Block{}, errors.New("a block with no repair time is a Poisson source at its rate, with no units' times to sweep: give the source a frequency with a range instead")
		}
	}
	d, err := dist.NewConstant(rate)
	if err != nil {
		return Block{}, errors.New("block rate: " + err.Error())
	}
	s.Frequency = d
	return b, nil
}
