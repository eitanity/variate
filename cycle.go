package variate

import (
	"encoding/binary"
	"math"
)

const (
	// maxJointPeriods caps the periods walked to find the schedules'
	// joint cycle, and maxJointPoints the distinct points kept from it:
	// one schedule of the longest length always fits.
	maxJointPeriods = 1 << 20
	maxJointPoints  = maxScheduleLength
)

// jointCycle is the schedules' values over their combined cycle: each
// distinct tuple of values a period sees, with its share of the cycle.
// A run steps every schedule by the same period, so the schedules move
// together and the mixtures take them as one quantity, not as
// independent draws. over is set when the cycle is too long to walk or
// has too many points.
type jointCycle struct {
	pools  []int32
	points [][]float64 // [point][schedule]
	weight []float64
	over   bool
}

// newJointCycle walks the combined cycle of the schedules, keeping the
// points in the order their first period reaches them.
func newJointCycle(scs []schedule) jointCycle {
	var c jointCycle
	if len(scs) == 0 {
		return c
	}
	n := int64(1)
	for _, sc := range scs {
		c.pools = append(c.pools, sc.pool)
		k := int64(len(sc.values))
		n = n / int64(gcd(int(n), int(k))) * k
		if n > maxJointPeriods {
			c.over = true
			return c
		}
	}
	seen := map[string]int{}
	key := make([]byte, 0, 8*len(scs))
	// bound: the combined cycle, at most maxJointPeriods
	for p := int64(0); p < n; p++ {
		key = key[:0]
		for _, sc := range scs {
			key = binary.LittleEndian.AppendUint64(key, math.Float64bits(sc.at(p)))
		}
		if i, ok := seen[string(key)]; ok {
			c.weight[i]++
			continue
		}
		if len(c.points) == maxJointPoints {
			c.over = true
			return c
		}
		seen[string(key)] = len(c.points)
		pt := make([]float64, len(scs))
		for s, sc := range scs {
			pt[s] = sc.at(p)
		}
		c.points = append(c.points, pt)
		c.weight = append(c.weight, 1)
	}
	for i := range c.weight {
		c.weight[i] /= float64(n)
	}
	return c
}

// quantity is the cycle as a period-scoped quantity, its nodes the
// indices of its points.
func (c *jointCycle) quantity() thetaQuantity {
	q := thetaQuantity{pool: noIndex, fn: noIndex, factor: true, cycle: c}
	if c.over {
		return q
	}
	q.nodes = make([]float64, len(c.points))
	for i := range q.nodes {
		q.nodes[i] = float64(i)
	}
	q.weight = append([]float64(nil), c.weight...)
	return q
}

// scheduleCross is E[a·b] over the combined cycle of two schedules. With
// g the greatest common divisor of their lengths, a period's places in
// the two cycles agree modulo g and are otherwise independent, so the
// moment is the mean over the g residues of the product of each
// schedule's mean on that residue. A schedule with itself gives E[a²].
func scheduleCross(a, b []float64) float64 {
	g := gcd(len(a), len(b))
	ra, rb := residueMeans(a, g), residueMeans(b, g)
	s := 0.0
	for r := range ra {
		s += ra[r] * rb[r]
	}
	return s / float64(g)
}

// residueMeans is the mean of the values at each place modulo g.
func residueMeans(v []float64, g int) []float64 {
	out := make([]float64, g)
	for i, x := range v {
		out[i%g] += x
	}
	per := float64(len(v) / g)
	for r := range out {
		out[r] /= per
	}
	return out
}

// scheduleRatios is E[a·b]/(E[a]·E[b]) for every pair of schedules: the
// factor two sources' joint count moment carries over the product of
// their means. 1 where a mean is 0, the sources then never firing.
func scheduleRatios(scs []schedule) [][]float64 {
	out := make([][]float64, len(scs))
	for i, a := range scs {
		out[i] = make([]float64, len(scs))
		for j, b := range scs {
			ma, mb := meanOf(a.values), meanOf(b.values)
			out[i][j] = 1
			if ma*mb != 0 {
				out[i][j] = scheduleCross(a.values, b.values) / (ma * mb)
			}
		}
	}
	return out
}

func meanOf(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

// scheduleIndex is the schedule whose pool is p.
func (m *Compiled) scheduleIndex(p int32) int {
	for i, sc := range m.schedules {
		if sc.pool == p {
			return i
		}
	}
	return -1
}
