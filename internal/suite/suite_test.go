package suite

import "os"

// The long statistical suite and the race-detector run are selected by
// environment variables set by the make targets, not by build tags, so
// every analyser sees every file in every run.

// longSuite reports whether the long statistical suite was asked for.
func longSuite() bool { return os.Getenv("VARIATE_STAT") == "1" }

// calibrationSeeds is the seed count: the fast suite's, or the long one's.
var calibrationSeeds = seedsFor(longSuite())

func seedsFor(long bool) int {
	if long {
		return 1000
	}
	return 60
}
