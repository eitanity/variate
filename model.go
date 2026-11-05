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
	Range          *InputRange // how uncertain Model is, for a sweep; nil = not swept
}

// Outcome is one branch of a node's selection.
type Outcome struct {
	To            string
	P             dist.Distribution // nil = remainder; Constant for fixed; otherwise drawn per period
	Holding       dist.Continuous   // nil = no delay
	HoldingScope  Scope
	OutageHolding bool        // hold for the time the reliability block that fired the round stays down
	PRange        *InputRange // how uncertain P is, for a sweep, as probabilities; nil = not swept
	HoldingRange  *InputRange // how uncertain Holding is; nil = not swept
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
	NextRange       *InputRange // how uncertain NextHolding is; nil = not swept
	Threshold       *Threshold  // a branch on one function's value, in place of Outcomes and Next; nil = none
	Trigger         *Trigger    // the node is entered once at a period's end when the period's sum of its inputs is above a level; nil = none
}

// Threshold sends a round one way when the value a node's function
// draws is above Over and the other way when it is at or below it. The
// value is drawn once and booked as usual; the round branches on it, so
// the branch and the value are as dependent as they are in the world.
type Threshold struct {
	Function     string  // the name of the node's OnEntry function whose value is tested
	Over         float64 // above is strictly greater than Over
	Above, Below string  // the next node on each side
}

// Trigger enters its node once, at the end of a period, when the sum
// of the values booked at its inputs by that period's events, or the
// count of that period's entries into its Entries nodes, is above Over.
// A booking counts toward the period its event arrived in, whatever
// period a holding time lands it in, and the walk the trigger starts
// books in that same period. No outcome leads into a trigger's node: it
// is entered only by firing. Its inputs may be reached by other
// triggers' walks, which then fire first in the period; a trigger whose
// sum its own firing feeds, directly or through others, is a cycle.
type Trigger struct {
	Over    float64        // fires when the sum is strictly greater
	Inputs  []TriggerInput // the values the sum adds up
	Entries []string       // or the nodes whose entries it counts, booking nothing
}

// TriggerInput is one term of a trigger's sum: one function of a node,
// or every function of it when Function is empty.
type TriggerInput struct {
	Node, Function string
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
	FrequencyRange  *InputRange       // how uncertain Frequency is, for a sweep; nil = not swept
	RepairRange     *InputRange       // how uncertain Repair is; nil = not swept
}

// Schedule is a cycle of multipliers on rates: period p uses the value
// at p modulo the cycle's length, so a rate can follow a season or a
// shift pattern. Over whole cycles every value is seen equally often,
// and every schedule steps with the same period, so the analytics take
// the schedules jointly over their combined cycle: a period drawn
// uniformly from it.
type Schedule struct {
	ID     string
	Values []float64   // finite, not negative
	Range  *InputRange // how uncertain the values' mean is, for a sweep; nil = not swept
}

// Factor is a multiplier on rates, drawn once per period and shared by
// every source naming it: a common shock, which makes the sources'
// counts dependent and the aggregate tail heavier than independence
// says.
type Factor struct {
	ID    string
	Model dist.Distribution // support in [0, ∞)
	Range *InputRange       // how uncertain Model is, for a sweep; nil = not swept
}

// Graph is a model: sources feeding a DAG of nodes.
type Graph struct {
	ID, Description string
	Sources         []Source
	Nodes           []Node
	Factors         []Factor
	Schedules       []Schedule
	Layers          []Layer // functions of the period total reported beside it
	Phased          Phase   // set by Phase: the place in the cycle every schedule is fixed at
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
	TransformBytes   int64   // what a transform pass may allocate; 0 = the default
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
