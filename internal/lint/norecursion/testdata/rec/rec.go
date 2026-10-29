package rec

func Fact(n int) int {
	if n <= 1 {
		return 1
	}
	return n * Fact(n-1)
}

func Ping(n int) int { return Pong(n) }
func Pong(n int) int {
	if n == 0 {
		return 0
	}
	return Ping(n - 1)
}

func Fine(xs []int) int {
	s := 0
	for _, x := range xs {
		s += x
	}
	return s
}
