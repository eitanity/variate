package config

import (
	"math"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// slot says where a distribution is used, which decides which modifiers
// it may carry and how a unit scales it.
type slot int

const (
	slotValue slot = iota
	slotRate
	slotHolding
	slotFrequency
	slotProbability
	slotFactor
)

func (s slot) String() string {
	return [...]string{"value", "rate", "holding time", "frequency", "probability", "factor"}[s]
}

// Family keys that the code names more than once.
const (
	famConstant    = "constant"
	famUniform     = "uniform"
	famLogUniform  = "loguniform"
	famTriangular  = "triangular"
	famPERT        = "pert"
	famLogPERT     = "logpert"
	famNormal      = "normal"
	famTruncNormal = "truncnormal"
	famLognormal   = "lognormal"
	famExponential = "exponential"
	famGamma       = "gamma"
	famBeta        = "beta"
	famWeibull     = "weibull"
	famPareto      = "pareto"
	famGPD         = "gpd"
	famBernoulli   = "bernoulli"
	famBinomial    = "binomial"
	famPoisson     = "poisson"
	famNegBin      = "negbin"
	famGeometric   = "geometric"
	famCategorical = "categorical"
	famEmpirical   = "empirical"
	unitPeriods    = "periods"
)

// families lists the family keys. The modifiers that may sit beside one
// are unit, t, truncate, cap and scope.
func families() []string {
	return []string{famConstant, famUniform, famLogUniform, famTriangular, famPERT, famLogPERT, famNormal,
		famTruncNormal, famLognormal, famExponential, famGamma, famBeta, famWeibull, famPareto, famGPD,
		famBernoulli, famBinomial, famPoisson, famNegBin, famGeometric, famCategorical, famEmpirical}
}

// distSpec is a distribution map taken apart: the family and its value
// node, and the modifiers.
type distSpec struct {
	family   string
	value    node
	unit     string
	t        float64
	hasT     bool
	truncate node
	cap      node
	scope    variate.Scope
	hasScope bool
	hasTrunc bool
	hasCap   bool
}

// parseDist reads a distribution: a bare number is a constant; a map has
// exactly one family key with optional modifiers. The entries of other
// keys (a function's name, an outcome's target) are taken from entries
// and ignored here; ignore lists them.
func (l *loader) parseDist(x node, s slot, ignore ...string) (dist.Distribution, variate.Scope, error) {
	if x.isScalar() {
		c, err := x.float()
		if err != nil {
			return nil, variate.PerEvent, err
		}
		d, err := dist.NewConstant(c)
		if err != nil {
			return nil, variate.PerEvent, x.errorf("%v", err)
		}
		return d, variate.PerEvent, nil
	}
	spec, err := l.takeApart(x, ignore)
	if err != nil {
		return nil, variate.PerEvent, err
	}
	if err := spec.checkSlot(x, s); err != nil {
		return nil, variate.PerEvent, err
	}
	scale, err := l.scale(x, spec.unit, s)
	if err != nil {
		return nil, variate.PerEvent, err
	}
	d, err := l.build(spec, scale)
	if err != nil {
		return nil, variate.PerEvent, err
	}
	d, err = spec.modify(d, scale)
	if err != nil {
		return nil, variate.PerEvent, err
	}
	return d, spec.scope, nil
}

// takeApart splits a map into its family and modifiers.
func (l *loader) takeApart(x node, ignore []string) (*distSpec, error) {
	es, err := x.entries()
	if err != nil {
		return nil, err
	}
	spec := &distSpec{}
	for _, e := range es {
		switch {
		case contains(ignore, e.key):
		case contains(families(), e.key):
			if spec.family != "" {
				return nil, x.errorf("both %q and %q name a family", spec.family, e.key)
			}
			spec.family, spec.value = e.key, e.value
		case e.key == "unit":
			if spec.unit, err = e.value.oneOf("hours", "days", "weeks", unitPeriods); err != nil {
				return nil, err
			}
		case e.key == "t":
			if spec.t, err = e.value.float(); err != nil {
				return nil, err
			}
			spec.hasT = true
		case e.key == "truncate":
			spec.truncate, spec.hasTrunc = e.value, true
		case e.key == "cap":
			spec.cap, spec.hasCap = e.value, true
		case e.key == "scope":
			sc, err := e.value.oneOf("event", "period")
			if err != nil {
				return nil, err
			}
			spec.scope, spec.hasScope = variate.PerEvent, true
			if sc == "period" {
				spec.scope = variate.PerPeriod
			}
		default:
			return nil, x.errorf("unknown key %q at line %d", e.key, e.line)
		}
	}
	if spec.family == "" {
		return nil, x.errorf("no distribution family named")
	}
	return spec, nil
}

// checkSlot rejects modifiers the slot does not take.
func (spec *distSpec) checkSlot(x node, s slot) error {
	switch {
	case spec.unit != "" && (s == slotValue || s == slotProbability || s == slotFactor):
		return x.errorf("unit is not allowed on a %s", s)
	case spec.hasScope && (s == slotProbability || s == slotFrequency || s == slotRate || s == slotFactor):
		if s == slotProbability {
			return x.errorf("a probability is always drawn per period; write the number for a fixed one")
		}
		return x.errorf("scope is not allowed on a %s", s)
	case spec.hasT && spec.family != famPERT && spec.family != famLogPERT:
		return x.errorf("t applies to pert and logpert only")
	}
	return nil
}

// scale is the factor a unit applies to the distribution's values: a
// duration in the unit becomes periods; a rate per unit becomes a rate
// per period.
func (l *loader) scale(x node, unit string, s slot) (float64, error) {
	if unit == "" || unit == unitPeriods {
		return 1, nil
	}
	if l.periodHours == 0 {
		return 0, x.errorf("unit: %s used but period: not declared", unit)
	}
	hours := unitHours(unit)
	if s == slotHolding {
		return hours / l.periodHours, nil
	}
	return l.periodHours / hours, nil
}

// build constructs the family with its parameters scaled by s.
func (l *loader) build(spec *distSpec, s float64) (dist.Distribution, error) {
	v := spec.value
	if s != 1 && !scales(spec.family) {
		return nil, v.errorf("unit is not allowed on %s", spec.family)
	}
	d, err := l.construct(spec, s)
	if err != nil {
		return nil, err
	}
	return d, nil
}

// scales reports whether a family is a scale family.
func scales(family string) bool {
	switch family {
	case famBeta, famBernoulli, famBinomial, famPoisson, famNegBin, famGeometric:
		return false
	}
	return true
}

func (l *loader) construct(spec *distSpec, s float64) (dist.Distribution, error) {
	v := spec.value
	t := float64(dist.DefaultPERTShape)
	if spec.hasT {
		t = spec.t
	}
	var d dist.Distribution
	var err error
	switch spec.family {
	case famConstant:
		d, err = wrap1(v, s, dist.NewConstant)
	case famUniform:
		d, err = wrap2(v, s, dist.NewUniform)
	case famLogUniform:
		d, err = wrap2(v, s, dist.NewLogUniform)
	case famTriangular:
		d, err = wrap3(v, s, dist.NewTriangular)
	case famPERT:
		d, err = wrap3(v, s, func(a, b, c float64) (dist.PERT, error) { return dist.NewPERT(a, b, c, t) })
	case famLogPERT:
		d, err = wrap3(v, s, func(a, b, c float64) (dist.LogPERT, error) { return dist.NewLogPERT(a, b, c, t) })
	case famNormal:
		d, err = named2(v, "mean", "sd", s, s, dist.NewNormal)
	case famTruncNormal:
		d, err = named4(v, "mean", "sd", "lo", "hi", s, dist.NewTruncatedNormal)
	case famLognormal:
		d, err = lognormal(v, s)
	case famExponential:
		d, err = exponential(v, s)
	case famGamma:
		d, err = gamma(v, s)
	case famBeta:
		d, err = beta(v)
	case famWeibull:
		d, err = named2(v, "shape", "scale", 1, s, dist.NewWeibull)
	case famPareto:
		d, err = named2(v, "min", "alpha", s, 1, dist.NewPareto)
	case famGPD:
		d, err = named3(v, "loc", "scale", "shape", s, s, 1, dist.NewGPD)
	case famBernoulli:
		d, err = wrap1(v, 1, dist.NewBernoulli)
	case famBinomial:
		d, err = named2(v, "n", "p", 1, 1, dist.NewBinomial)
	case famPoisson:
		d, err = wrap1(v, 1, dist.NewPoisson)
	case famNegBin:
		d, err = negbin(v)
	case famGeometric:
		d, err = wrap1(v, 1, dist.NewGeometric)
	case famCategorical:
		d, err = categorical(v, s)
	case famEmpirical:
		d, err = l.empirical(v, s)
	}
	if err != nil {
		return nil, err
	}
	return d, nil
}

// wrap1, wrap2, wrap3 call a positional constructor on a scalar or a
// list of 2 or 3 numbers, scaled.
func wrap1[D dist.Distribution](v node, s float64, ctor func(float64) (D, error)) (dist.Distribution, error) {
	a, err := v.float()
	if err != nil {
		return nil, err
	}
	d, err := ctor(a * s)
	if err != nil {
		return nil, v.errorf("%v", err)
	}
	return d, nil
}

func wrap2[D dist.Distribution](v node, s float64, ctor func(a, b float64) (D, error)) (dist.Distribution, error) {
	p, err := v.floats(2)
	if err != nil {
		return nil, err
	}
	d, err := ctor(p[0]*s, p[1]*s)
	if err != nil {
		return nil, v.errorf("%v", err)
	}
	return d, nil
}

func wrap3[D dist.Distribution](v node, s float64, ctor func(a, b, c float64) (D, error)) (dist.Distribution, error) {
	p, err := v.floats(3)
	if err != nil {
		return nil, err
	}
	d, err := ctor(p[0]*s, p[1]*s, p[2]*s)
	if err != nil {
		return nil, v.errorf("%v", err)
	}
	return d, nil
}

// named reads a map with exactly the given keys, all numbers.
func named(v node, keys ...string) ([]float64, error) {
	m, err := v.mapping(keys...)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(keys))
	for i, k := range keys {
		x, ok := m[k]
		if !ok {
			return nil, v.errorf("missing %q", k)
		}
		if out[i], err = x.float(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func named2[D dist.Distribution](v node, k1, k2 string, s1, s2 float64, ctor func(a, b float64) (D, error)) (dist.Distribution, error) {
	p, err := named(v, k1, k2)
	if err != nil {
		return nil, err
	}
	d, err := ctor(p[0]*s1, p[1]*s2)
	if err != nil {
		return nil, v.errorf("%v", err)
	}
	return d, nil
}

func named3[D dist.Distribution](v node, k1, k2, k3 string, s1, s2, s3 float64, ctor func(a, b, c float64) (D, error)) (dist.Distribution, error) {
	p, err := named(v, k1, k2, k3)
	if err != nil {
		return nil, err
	}
	d, err := ctor(p[0]*s1, p[1]*s2, p[2]*s3)
	if err != nil {
		return nil, v.errorf("%v", err)
	}
	return d, nil
}

func named4[D dist.Distribution](v node, k1, k2, k3, k4 string, s float64, ctor func(a, b, c, d float64) (D, error)) (dist.Distribution, error) {
	p, err := named(v, k1, k2, k3, k4)
	if err != nil {
		return nil, err
	}
	d, err := ctor(p[0]*s, p[1]*s, p[2]*s, p[3]*s)
	if err != nil {
		return nil, v.errorf("%v", err)
	}
	return d, nil
}

// keysOf returns the sorted key set of a map node, for choosing a
// parameterisation.
func keysOf(v node) (map[string]node, error) {
	es, err := v.entries()
	if err != nil {
		return nil, err
	}
	m := make(map[string]node, len(es))
	for _, e := range es {
		m[e.key] = e.value
	}
	return m, nil
}

func has(m map[string]node, keys ...string) bool {
	if len(m) != len(keys) {
		return false
	}
	for _, k := range keys {
		if _, ok := m[k]; !ok {
			return false
		}
	}
	return true
}

func lognormal(v node, s float64) (dist.Distribution, error) {
	m, err := keysOf(v)
	if err != nil {
		return nil, err
	}
	switch {
	case has(m, "mu", "sigma"):
		return named2(v, "mu", "sigma", 1, 1, func(mu, sigma float64) (dist.Lognormal, error) {
			return dist.NewLognormal(mu+math.Log(s), sigma)
		})
	case has(m, "median", "p90"):
		return named2(v, "median", "p90", s, s, dist.NewLognormalFromMedianP90)
	case has(m, "mean", "sd"):
		return named2(v, "mean", "sd", s, s, dist.NewLognormalFromMeanSD)
	}
	return nil, v.errorf("lognormal takes {mu, sigma}, {median, p90} or {mean, sd}")
}

func exponential(v node, s float64) (dist.Distribution, error) {
	m, err := keysOf(v)
	if err != nil {
		return nil, err
	}
	switch {
	case has(m, "mean"):
		return named2(v, "mean", "mean", s, s, func(a, _ float64) (dist.Exponential, error) { return dist.NewExponential(a) })
	case has(m, "rate"):
		return named2(v, "rate", "rate", 1/s, 1/s, func(a, _ float64) (dist.Exponential, error) { return dist.NewExponentialFromRate(a) })
	}
	return nil, v.errorf("exponential takes {mean} or {rate}")
}

func gamma(v node, s float64) (dist.Distribution, error) {
	m, err := keysOf(v)
	if err != nil {
		return nil, err
	}
	switch {
	case has(m, "shape", "scale"):
		return named2(v, "shape", "scale", 1, s, dist.NewGamma)
	case has(m, "shape", "rate"):
		return named2(v, "shape", "rate", 1, 1/s, dist.NewGammaFromShapeRate)
	case has(m, "mean", "sd"):
		return named2(v, "mean", "sd", s, s, dist.NewGammaFromMeanSD)
	}
	return nil, v.errorf("gamma takes {shape, scale}, {shape, rate} or {mean, sd}")
}

func beta(v node) (dist.Distribution, error) {
	if v.isSequence() {
		return wrap2(v, 1, dist.NewBeta)
	}
	m, err := keysOf(v)
	if err != nil {
		return nil, err
	}
	if has(m, "mean", "sd") {
		return named2(v, "mean", "sd", 1, 1, dist.NewBetaFromMeanSD)
	}
	return nil, v.errorf("beta takes [alpha, beta] or {mean, sd}")
}

func negbin(v node) (dist.Distribution, error) {
	m, err := keysOf(v)
	if err != nil {
		return nil, err
	}
	switch {
	case has(m, "r", "p"):
		return named2(v, "r", "p", 1, 1, dist.NewNegativeBinomial)
	case has(m, "mean", "size"):
		return named2(v, "mean", "size", 1, 1, dist.NewNegativeBinomialFromMeanSize)
	}
	return nil, v.errorf("negbin takes {r, p} or {mean, size}")
}

func categorical(v node, s float64) (dist.Distribution, error) {
	m, err := v.mapping("values", "p")
	if err != nil {
		return nil, err
	}
	vals, ok := m["values"]
	if !ok {
		return nil, v.errorf("missing values")
	}
	ps, ok := m["p"]
	if !ok {
		return nil, v.errorf("missing p")
	}
	values, err := vals.floats(0)
	if err != nil {
		return nil, err
	}
	probs, err := ps.floats(len(values))
	if err != nil {
		return nil, err
	}
	for i := range values {
		values[i] *= s
	}
	d, err := dist.NewCategorical(values, probs)
	if err != nil {
		return nil, v.errorf("%v", err)
	}
	return d, nil
}

// modify applies truncate and cap, whose bounds are in the family's unit.
func (spec *distSpec) modify(d dist.Distribution, s float64) (dist.Distribution, error) {
	if spec.hasTrunc {
		b, err := spec.truncate.floats(2)
		if err != nil {
			return nil, err
		}
		t, err := dist.NewTruncated(d, b[0]*s, b[1]*s)
		if err != nil {
			return nil, spec.truncate.errorf("%v", err)
		}
		d = t
	}
	if spec.hasCap {
		c, err := spec.cap.float()
		if err != nil {
			return nil, err
		}
		cd, err := dist.NewCapped(d, c*s)
		if err != nil {
			return nil, spec.cap.errorf("%v", err)
		}
		d = cd
	}
	return d, nil
}
