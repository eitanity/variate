package stats

func anys(v any) int { // want "any in the engine"
	var e interface{} // want "empty interface in the engine"
	_ = e
	switch v.(type) { // want "type switch outside config"
	case int:
		return 1
	}
	if x, ok := v.(int); ok { // want "type assertion outside config"
		return x
	}
	return 0
}
