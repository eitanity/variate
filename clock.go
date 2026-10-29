package variate

import "time"

// clock is the one place the engine reads the wall clock or sleeps, so
// that pacing can be tested with a fake.
type clock interface {
	Now() time.Time
	Sleep(d time.Duration)
}

type realClock struct{}

func (realClock) Now() time.Time        { return time.Now() }
func (realClock) Sleep(d time.Duration) { time.Sleep(d) }
