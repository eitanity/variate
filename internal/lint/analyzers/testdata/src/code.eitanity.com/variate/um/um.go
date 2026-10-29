package um

import "sort"

func maps(m map[string]int) int {
	s := 0
	for _, v := range m { // want "range over a map without a stated reason"
		s += v
	}
	// unordered: each key fills its own slot
	for k, v := range m {
		_ = k
		s += v
	}
	keys := make([]string, 0, len(m))
	// unordered: the keys are sorted before use
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		s += m[k]
	}
	return s
}
