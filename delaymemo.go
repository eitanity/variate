package variate

import "math"

// The delay figures of a model conditioned on its one value factor at f
// are a quadratic in f where the factor scales values alone: a
// booking's second moments are f², f or 1 times the factor's own. Three
// points' figures fix the quadratic, and a fourth computed one tests it
// to delayMemoTol, the figures' own settling; once it holds, the other
// points' figures are read from it, and where it does not every point is
// computed.

// delayMemoTol is the relative difference by which a point's figures
// and the quadratic through three others agree.
const delayMemoTol = 1e-9

// delayPoint is one factor value's delay figures.
type delayPoint struct {
	f                  float64
	perPeriod, longRun map[delayKey]float64
	profiles           map[delayKey][]float64
}

// delayMemo keeps a conditioning's delay figures by factor value.
type delayMemo struct {
	points []delayPoint
	// quad is 1 once a fourth point fits the quadratic through the first
	// three, −1 once one does not, 0 before.
	quad int
}

// delaysAt is the delay figures at factor value f, computed or, once the
// quadratic holds, read from it; computed when f is NaN or rc is nil.
func (rc *ruleCache) delaysAt(f float64, compute func() (map[delayKey]float64, map[delayKey]float64, map[delayKey][]float64)) (map[delayKey]float64, map[delayKey]float64, map[delayKey][]float64) {
	if rc == nil || math.IsNaN(f) {
		return compute()
	}
	dm := &rc.delays
	for _, p := range dm.points {
		if p.f == f {
			return p.perPeriod, p.longRun, p.profiles
		}
	}
	if dm.quad == 1 {
		q := dm.quadAt(f)
		return q.perPeriod, q.longRun, q.profiles
	}
	pp, lr, pr := compute()
	got := delayPoint{f: f, perPeriod: pp, longRun: lr, profiles: pr}
	if dm.quad == 0 && len(dm.points) == 3 {
		dm.quad = -1
		if fits(dm.quadAt(f), got) {
			dm.quad = 1
		}
	}
	if len(dm.points) < 3 {
		dm.points = append(dm.points, got)
	}
	return pp, lr, pr
}

// quadAt is the quadratic through the first three points at f, by
// Lagrange's weights.
func (dm *delayMemo) quadAt(f float64) delayPoint {
	var l [3]float64
	for i := range l {
		l[i] = 1
		for j := range l {
			if j != i {
				l[i] *= (f - dm.points[j].f) / (dm.points[i].f - dm.points[j].f)
			}
		}
	}
	out := delayPoint{f: f, perPeriod: map[delayKey]float64{}, longRun: map[delayKey]float64{}}
	// unordered: each key's sum is its own
	for k := range dm.points[0].perPeriod {
		for i, p := range dm.points {
			out.perPeriod[k] += l[i] * p.perPeriod[k]
			out.longRun[k] += l[i] * p.longRun[k]
		}
	}
	if dm.points[0].profiles != nil {
		out.profiles = map[delayKey][]float64{}
		// unordered: each key's profile is its own
		for k, v := range dm.points[0].profiles {
			s := make([]float64, len(v))
			for i, p := range dm.points {
				for d, x := range p.profiles[k] {
					s[d] += l[i] * x
				}
			}
			out.profiles[k] = s
		}
	}
	return out
}

// fits reports whether a point's figures agree with the quadratic's to
// delayMemoTol, NaN in either failing.
func fits(q, got delayPoint) bool {
	near := func(a, b float64) bool { return math.Abs(a-b) <= delayMemoTol*math.Max(math.Abs(a), math.Abs(b)) }
	if len(q.perPeriod) != len(got.perPeriod) || (q.profiles == nil) != (got.profiles == nil) {
		return false
	}
	// unordered: every key is tested
	for k, v := range got.perPeriod {
		if !near(q.perPeriod[k], v) || !near(q.longRun[k], got.longRun[k]) {
			return false
		}
	}
	// unordered: every key is tested
	for k, v := range got.profiles {
		w := q.profiles[k]
		if len(w) != len(v) {
			return false
		}
		for d := range v {
			if !near(w[d], v[d]) {
				return false
			}
		}
	}
	return true
}
