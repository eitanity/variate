package config

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"go.yaml.in/yaml/v3"
)

// node wraps a YAML node with the key path that reached it, for errors.
type node struct {
	n    *yaml.Node
	path string
}

func (x node) line() int { return x.n.Line }

// errorf formats an error that names the path and line.
func (x node) errorf(format string, args ...any) error {
	return fmt.Errorf("%s (line %d): %s", x.path, x.line(), fmt.Sprintf(format, args...))
}

// deref follows aliases, which the decoder leaves in place.
func deref(n *yaml.Node) *yaml.Node {
	// bound: alias chains are finite in a parsed document
	for i := 0; i < 64 && n.Kind == yaml.AliasNode && n.Alias != nil; i++ {
		n = n.Alias
	}
	return n
}

func (x node) kind() yaml.Kind { return deref(x.n).Kind }

func (x node) isSequence() bool { return x.kind() == yaml.SequenceNode }
func (x node) isScalar() bool   { return x.kind() == yaml.ScalarNode }

// entry is one key/value pair of a mapping.
type entry struct {
	key   string
	value node
	line  int
}

// entries lists a mapping's pairs in order; duplicate keys are an error.
func (x node) entries() ([]entry, error) {
	n := deref(x.n)
	if n.Kind != yaml.MappingNode {
		return nil, x.errorf("expected a map")
	}
	out := make([]entry, 0, len(n.Content)/2)
	seen := map[string]bool{}
	for i := 0; i+1 < len(n.Content); i += 2 {
		k := deref(n.Content[i])
		if k.Kind != yaml.ScalarNode {
			return nil, x.errorf("map key at line %d is not a scalar", k.Line)
		}
		if seen[k.Value] {
			return nil, x.errorf("duplicate key %q at line %d", k.Value, k.Line)
		}
		seen[k.Value] = true
		out = append(out, entry{key: k.Value, value: node{n: n.Content[i+1], path: x.path + "." + k.Value}, line: k.Line})
	}
	return out, nil
}

// mapping reads a mapping whose keys must all be in allowed, returning
// the values by key.
func (x node) mapping(allowed ...string) (map[string]node, error) {
	es, err := x.entries()
	if err != nil {
		return nil, err
	}
	out := make(map[string]node, len(es))
	for _, e := range es {
		if !contains(allowed, e.key) {
			return nil, x.errorf("unknown key %q at line %d", e.key, e.line)
		}
		out[e.key] = e.value
	}
	return out, nil
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// items lists a sequence's elements.
func (x node) items() ([]node, error) {
	n := deref(x.n)
	if n.Kind != yaml.SequenceNode {
		return nil, x.errorf("expected a list")
	}
	out := make([]node, len(n.Content))
	for i, c := range n.Content {
		out[i] = node{n: c, path: x.path + "[" + strconv.Itoa(i) + "]"}
	}
	return out, nil
}

// scalar returns the scalar's text.
func (x node) scalar() (string, error) {
	n := deref(x.n)
	if n.Kind != yaml.ScalarNode {
		return "", x.errorf("expected a value")
	}
	return n.Value, nil
}

// str returns a non-empty string.
func (x node) str() (string, error) {
	s, err := x.scalar()
	if err != nil {
		return "", err
	}
	if s == "" {
		return "", x.errorf("empty value")
	}
	return s, nil
}

// float parses a finite or infinite number; .inf and -.inf are allowed.
func (x node) float() (float64, error) {
	s, err := x.scalar()
	if err != nil {
		return 0, err
	}
	switch strings.ToLower(s) {
	case ".inf", "+.inf", "inf", "+inf":
		return math.Inf(1), nil
	case "-.inf", "-inf":
		return math.Inf(-1), nil
	}
	v, err := strconv.ParseFloat(strings.ReplaceAll(s, "_", ""), 64)
	if err != nil || math.IsNaN(v) {
		return 0, x.errorf("%q is not a number", s)
	}
	return v, nil
}

// floats parses a list of numbers of the given length (0 = any).
func (x node) floats(want int) ([]float64, error) {
	items, err := x.items()
	if err != nil {
		return nil, err
	}
	if want > 0 && len(items) != want {
		return nil, x.errorf("expected %d numbers, got %d", want, len(items))
	}
	out := make([]float64, len(items))
	for i, it := range items {
		if out[i], err = it.float(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// integer parses a whole number within [lo, hi].
func (x node) integer(lo, hi int64) (int64, error) {
	s, err := x.scalar()
	if err != nil {
		return 0, err
	}
	v, err := strconv.ParseInt(strings.ReplaceAll(s, "_", ""), 10, 64)
	if err != nil {
		return 0, x.errorf("%q is not a whole number", s)
	}
	if v < lo || v > hi {
		return 0, x.errorf("%d is outside %d..%d", v, lo, hi)
	}
	return v, nil
}

// boolean parses true or false.
func (x node) boolean() (bool, error) {
	s, err := x.scalar()
	if err != nil {
		return false, err
	}
	switch s {
	case "true", "yes", "on":
		return true, nil
	case "false", "no", "off":
		return false, nil
	}
	return false, x.errorf("%q is not true or false", s)
}

// oneOf parses a string from a fixed set.
func (x node) oneOf(allowed ...string) (string, error) {
	s, err := x.str()
	if err != nil {
		return "", err
	}
	if !contains(allowed, s) {
		return "", x.errorf("%q is not one of %s", s, strings.Join(allowed, ", "))
	}
	return s, nil
}

// clone deep-copies a node tree with an explicit stack, resolving
// aliases.
func clone(root *yaml.Node) *yaml.Node {
	type job struct {
		src *yaml.Node
		dst *yaml.Node
	}
	out := &yaml.Node{}
	stack := []job{{deref(root), out}}
	// bound: nodes in the document
	for len(stack) > 0 {
		j := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		*j.dst = *j.src
		j.dst.Alias = nil
		j.dst.Content = make([]*yaml.Node, len(j.src.Content))
		for i, ch := range j.src.Content {
			j.dst.Content[i] = &yaml.Node{}
			stack = append(stack, job{deref(ch), j.dst.Content[i]})
		}
	}
	return out
}

var errTooLarge = errors.New("model file is larger than 16 MB")

var errNotAPath = errors.New("not an override path")
