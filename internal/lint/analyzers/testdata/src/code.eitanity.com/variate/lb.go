package variate

func loops(xs []int, n int) int {
	s := 0
	for _, x := range xs {
		s += x
	}
	for i := 0; i < len(xs); i++ {
		s += xs[i]
	}
	for i := 0; i < 1000; i++ {
		s++
	}
	// bound: n is validated by Compile
	for i := 0; i < n; i++ {
		s++
	}
	for s > n { // want "loop without a stated bound"
		s--
	}
	for { // want "loop without a stated bound"
		break
	}
	return s
}
