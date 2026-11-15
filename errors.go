package variate

import (
	"errors"
	"math"
	"strconv"
)

// modelError describes a validation failure, naming its subject.
func modelError(kind, subject, msg string) error {
	return errors.New(kind + " " + strconv.Quote(subject) + ": " + msg)
}

func ftoa(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }

// figure formats a derived value for a message to six significant digits,
// so that the last-unit differences between architectures do not show.
func figure(v float64) string { return strconv.FormatFloat(v, 'g', 6, 64) }

func itoa(v int) string { return strconv.Itoa(v) }

func i64toa(v int64) string { return strconv.FormatInt(v, 10) }

// index narrows a count to the int32 the tables use; Compile refuses a
// graph that would exceed it before any table is built.
func index(n int) int32 {
	if n < 0 || n > math.MaxInt32 {
		return noIndex
	}
	return int32(n) //nolint:gosec // bounded by the check above
}

// Errors raised on the hot path are built here, off it.

// pendingError reports a pending list full at period p.
func pendingError(capacity int, p int64) error {
	return errors.New("more than " + itoa(capacity) + " bookings pending past period " + ftoa(float64(p)) + "; the list is sized at eight times the expected number in flight")
}

func invariantError(cats, srcs, total float64) error {
	return errors.New("invariant: breakdowns " + ftoa(cats) + " and " + ftoa(srcs) + " do not sum to the total " + ftoa(total))
}

// recordFileStage names the slot file in errors.
const recordFileStage = "record file"

// The two constructors below are kept out of line so that the error
// values are built here, off the hot path, only when an error occurs.

//go:noinline
func recordFileError(err error) error { return runError{stage: recordFileStage, err: err} }

//go:noinline
func observerError(err error) error { return runError{stage: "observer", err: err} }

//go:noinline
func eventsCapError(source string, p int64) error {
	return runError{stage: "run", err: errors.New("source " + source + " drew more than " + itoa(maxEventsPeriod) + " arrivals in period " + strconv.FormatInt(p, 10) + "; its rate is beyond what a period can hold")}
}

//go:noinline
func observerPanicError() error { return errors.New("observer panicked") }

//go:noinline
func scratchFullError(rounds int) error {
	return errors.New("round scratch full: more than " + itoa(rounds) + " matching rounds in one period; narrow the Watch or raise RoundScratch")
}

//go:noinline
func noRecordFileError() error { return errors.New("record buffer full and no record file") }

func queueFullError(queue string, capacity int) error {
	return errors.New("queue " + queue + ": more than " + itoa(capacity) + " rounds at once; the line is sized at twice the rounds expected to reach it in a task")
}

func givenCountError(entry string, n float64) error {
	return errors.New("start: " + entry + " drew " + figure(n) + " rounds, more than " + itoa(maxGiven) + " a repeat")
}

func drainError(periods int) error {
	return errors.New("the queues did not empty within " + itoa(periods) + " periods after the horizon")
}
