package variate

import (
	"errors"
	"time"

	"code.eitanity.com/variate/dist"
)

// Basis says how a function books its value: once per trigger, or as a
// rate multiplied by the holding time of the outcome taken.
type Basis uint8

const (
	OnEntry Basis = iota
	PerTime
)

// Scope says how widely one draw is shared: per event, or per period.
type Scope uint8

const (
	PerEvent Scope = iota
	PerPeriod
)

// Mode is a node's selection mode.
type Mode uint8

const (
	Absolute Mode = iota
	Sequential
)

// Function is a pdf a node applies on every trigger, booked to a category.
type Function struct {
	Name, Category string
	Model          dist.Distribution // value; for PerTime, a rate per period
	Basis          Basis
	Scope          Scope
}

// Outcome is one branch of a node's selection.
type Outcome struct {
	To            string
	P             dist.Distribution // nil = remainder; Constant for fixed; otherwise drawn per period
	Holding       dist.Continuous   // nil = no delay
	HoldingScope  Scope
	OutageHolding bool // hold for the time the reliability block that fired the round stays down
}

// Node is a state an event passes through.
type Node struct {
	ID, Description string
	Functions       []Function
	Mode            Mode
	Outcomes        []Outcome
	Next            string          // continuation without a selection; "" with no Outcomes = leaf
	NextOutage      bool            // the continuation holds for the block's outage (as OutageHolding)
	NextHolding     dist.Continuous // holding time on Next; nil = 0
	NextScope       Scope
}

// Source is an initiating event flow.
type Source struct {
	ID, Description string
	Frequency       dist.Distribution // rate per period
	To              string
	Every           int               // fire on every Every-th arrival of the stream; 0 or 1 = every arrival
	Repair          dist.Distribution // holding after each firing, in periods, during which the source is silent; nil = none
	Chains          []Chain           // a reliability diagram the engine simulates in place of Frequency; nil = none
	Factor          string            // a Factor's ID scaling the rate, drawn once per period and shared; "" = none
	Schedule        string            // a Schedule's ID scaling the rate by the period's place in a cycle; "" = none
}

// Schedule is a cycle of multipliers on rates: period p uses the value
// at p modulo the cycle's length, so a rate can follow a season or a
// shift pattern. Over whole cycles every value is seen equally often,
// so the analytics treat the schedule as a multiplier drawn uniformly
// from its values.
type Schedule struct {
	ID     string
	Values []float64 // finite, not negative
}

// Factor is a multiplier on rates, drawn once per period and shared by
// every source naming it: a common shock, which makes the sources'
// counts dependent and the aggregate tail heavier than independence
// says.
type Factor struct {
	ID    string
	Model dist.Distribution // support in [0, ∞)
}

// Graph is a model: sources feeding a DAG of nodes.
type Graph struct {
	ID, Description string
	Sources         []Source
	Nodes           []Node
	Factors         []Factor
	Schedules       []Schedule
	Layers          []Layer // functions of the period total reported beside it
}

// Config holds every run setting; each field's zero value selects the
// default.
type Config struct {
	Periods          int64
	AllowUnconverged bool    // run to the caps when the plan exceeds them
	Transform        bool    // compute the transform at compile; the tail criterion then counts periods beyond its quantile
	TransformGrid    int     // largest grid of that transform; 0 = the transform's default
	TransformTol     float64 // its tolerance; 0 = the default
	TransformWork    int64   // its work budget; 0 = the default
	TransformBytes   int64   // the grids a transform pass may hold; 0 = the default
	MinHits          uint64
	ContribRSE       float64
	TargetRSE        float64
	TailP            float64
	TailHits         uint64
	MaxPeriods       int64
	MaxSteps         uint64
	Batch            int
	HistRes          float64

	Seed         uint64
	Workers      int
	BlockPeriods int
	PoolSize     int
	Series       bool
	NoBreakdown  bool
	NodeHist     bool
	Spill        int
	MaxMemory    int64
	Observer     Observer
	Watch        Watch
	Pace         time.Duration
	RoundScratch int
	RecordBuffer int
}

// Watch selects which rounds an Observer receives: a round matches every
// non-empty field, and any entry within a list.
type Watch struct {
	Nodes    []string
	Leaves   []string
	MinValue float64
}

// Observer receives delivered rounds, one at a time, from one goroutine.
// Returning ErrStop ends the run; any other error fails it.
type Observer interface {
	Observe(r *Round) error
}

// ErrStop is returned by an Observer to end the run cleanly.
var ErrStop = errors.New("stop")

// Round is one delivered event with its whole path.
type Round struct {
	Seq     uint64
	Period  int64
	Source  int32
	Arrival float64
	End     float64
	Leaf    int32
	Value   float64
	Steps   []Step
}

// Step is one node visit on a round's path.
type Step struct {
	Node    int32
	Outcome int32 // −1 at the leaf
	T       float64
	Value   float64
}

// Warning is a non-fatal finding of Compile.
type Warning struct {
	Code    string
	Subject string
	Message string
}
