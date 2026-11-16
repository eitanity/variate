package config

import (
	"errors"
	"strconv"
	"strings"

	"code.eitanity.com/variate"
)

// ParseLevel reads a value of a quantity to count, as kind:name=x, or
// total=x for the total: a category, source, entries, contrib, layer or
// trigger by its name.
func ParseLevel(s string) (variate.Level, error) {
	q, xs, ok := strings.Cut(s, "=")
	if !ok {
		return variate.Level{}, errors.New("at " + s + ": write kind:name=value, as category:labour=40, or total=value")
	}
	x, err := strconv.ParseFloat(strings.TrimSpace(xs), 64)
	if err != nil {
		return variate.Level{}, errors.New("at " + s + ": the value " + xs + " is not a number")
	}
	kind, name, _ := strings.Cut(strings.TrimSpace(q), ":")
	return variate.Level{Kind: kind, Name: name, X: x}, nil
}
