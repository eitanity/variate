package config

import (
	"errors"
	"strconv"
	"strings"

	"code.eitanity.com/variate"
)

// ParseWhen reads a condition on a quantity, as kind:name>=b for the
// periods at or above b, or kind:name in [a,b) for those from a to below
// b, total for the total; a bound is a value, or 1/N for the level
// reached once in N periods.
func ParseWhen(s string) (variate.When, error) {
	usage := "when " + s + ": write kind:name>=bound or kind:name in [bound,bound), a bound a value or 1/N, as total>=1/100"
	q, b, ok := strings.Cut(s, ">=")
	if ok {
		from, err := parseBound(b)
		if err != nil {
			return variate.When{}, errors.New("when " + s + ": " + err.Error())
		}
		return whenOf(q, from, nil), nil
	}
	q, r, ok := strings.Cut(s, " in ")
	r = strings.TrimSpace(r)
	if !ok || !strings.HasPrefix(r, "[") || !strings.HasSuffix(r, ")") {
		return variate.When{}, errors.New(usage)
	}
	a, b, ok := strings.Cut(r[1:len(r)-1], ",")
	if !ok {
		return variate.When{}, errors.New(usage)
	}
	from, err := parseBound(a)
	if err != nil {
		return variate.When{}, errors.New("when " + s + ": " + err.Error())
	}
	to, err := parseBound(b)
	if err != nil {
		return variate.When{}, errors.New("when " + s + ": " + err.Error())
	}
	return whenOf(q, from, &to), nil
}

func whenOf(q string, from variate.Bound, to *variate.Bound) variate.When {
	kind, name, _ := strings.Cut(strings.TrimSpace(q), ":")
	return variate.When{Kind: kind, Name: name, From: from, To: to}
}

// parseBound reads a value, or 1/N as a return period.
func parseBound(s string) (variate.Bound, error) {
	s = strings.TrimSpace(s)
	if n, ok := strings.CutPrefix(s, "1/"); ok {
		x, err := strconv.ParseFloat(n, 64)
		if err != nil {
			return variate.Bound{}, errors.New("the return period " + s + " is not 1/N")
		}
		return variate.Bound{OneIn: x}, nil
	}
	x, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return variate.Bound{}, errors.New("the bound " + s + " is neither a number nor 1/N")
	}
	return variate.Bound{X: x}, nil
}
