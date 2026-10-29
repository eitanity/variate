package config

func walk(v any) int {
	switch x := v.(type) {
	case int:
		return x
	}
	if x, ok := v.(int); ok {
		return x
	}
	for v != nil {
		v = nil
	}
	return 0
}
