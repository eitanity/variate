package dist

import "os"

// nDraws is the sample size: the fast suite's, or the long one's when
// VARIATE_STAT=1 asks for it.
var nDraws = drawsFor(os.Getenv("VARIATE_STAT") == "1")

func drawsFor(long bool) int {
	if long {
		return 10_000_000
	}
	return 200_000
}
