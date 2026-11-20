# The model file

A Variate model is one YAML file: a period, optional run settings,
optional templates, and the graphs. Everything a run can do is in this
file; the command line only overrides settings.

```yaml
period: { days: 365 }        # how long one period is (needed for any time unit)
periods: 100000              # a setting: run this many periods (omit to let the plan decide)
graphs:
  - id: incidents
    sources:
      - { id: incident, frequency: { pert: [2, 5, 12] }, to: loss }
    nodes:
      - id: loss
        functions:
          - { name: cost, category: money, lognormal: { median: 1000, p90: 5000 } }
```

Run it with `variate run -f model.yaml`, see the plan with
`variate plan -f model.yaml`, or load it in the browser demo's YAML box.
Every key below is checked when the file is read; an unknown key, a
missing one or a value out of range is an error that names the key and
the line.

## Top level

| key | meaning |
|---|---|
| `phase` | run one place in the schedules' cycle, counted from 1; 0 or omitted is the whole cycle (see rates that vary by period). |
| `start` | where a run begins: omitted, the steady state, every period alike; `{ from: empty, horizon: n }`, a started run of n periods from an empty start, repeated (below). |
| `period` | the length of one period, as one of `{ hours: n }`, `{ days: n }`, `{ weeks: n }`. Needed when any distribution, holding time or block uses a time `unit`. |
| `graphs` | a list of graphs; sources and nodes of every graph form one model. |
| `templates` | reusable sub-graphs instanced inside graphs (below). |
| `factors` | shared multipliers on rates, each `{ id, <distribution> }` on [0, ∞), drawn once per period and shared by every source naming it: a common shock (below). |
| `schedules` | cycles of multipliers on rates, each `{ id, values: [...] }`; period p uses the value at p modulo the cycle's length: a season or a shift pattern (below). |
| `layers` | functions of the period total reported beside it, each `{ id, attachment, limit, share }`: an aggregate deductible, an aggregate limit or a reinsurance layer (below). |
| `totals` | sums of chosen categories reported beside the total, each `{ id, categories: [...], less: [...] }`, `less` optional (below). |
| `outside_total` | categories booked and reported but added to no total but a named one, such as hours beside money (below). |
| `discount` | in a started run, the categories whose values are discounted by the period they land in, `{ rate: r, categories: [...] }` or `{ factors: [...], categories: [...] }` (below). |
| `dependence` | groups of values one round draws that move together, each `{ functions: [...], rank: r }` (below). |
| settings | any run setting by its snake_case name, e.g. `periods`, `seed`, `transform` (table at the end). |

The file may hold one YAML document, without anchors, aliases or custom
tags, and no deeper than 32 levels. Files over 16 MB are refused.

## Graphs

```yaml
graphs:
  - id: site
    description: optional text
    sources: [ ... ]
    nodes: [ ... ]
    instances: [ ... ]      # copies of templates, optional
```

### Sources

A source is a stream of events entering the graph at a node. It has one
of three forms of rate.

```yaml
- { id: trip, frequency: { pert: [20, 60, 200] }, to: trip_check }
- { id: pallets, frequency: 20000, every: 500, to: filter_change }
- { id: press, frequency: 12, repair: { pert: [1, 3, 10], unit: days }, to: press_repair }
- id: plant_down
  block: { unit: hours, series: [ { kofn: { k: 2, n: 3, mtbf: 6000, mttr: 36 } }, { mtbf: 30000, mttr: 8 } ] }
  to: outage
```

| key | meaning |
|---|---|
| `id` | name, unique among sources. |
| `description` | optional text. |
| `frequency` | events per period: a number or a distribution with support in [0, ∞). An uncertain rate is drawn once per period and the count that period is Poisson at it (a gamma rate gives negative-binomial counts). With `unit`, a rate per hour, day or week. |
| `to` | the node the events enter. |
| `every` | fire on every n-th arrival of the stream only, for things that happen after n inputs; the count per period is exact and much less spread than Poisson. |
| `repair` | a holding time (distribution with `unit`) during which the source is silent after each firing: a unit that cannot fail again until repaired. Needs a constant `frequency`. The long-run rate is 1/(1/λ + mean repair). |
| `block` | a reliability block diagram in place of `frequency` (below). Cannot be combined with `every` or `repair`. |
| `factor` | the id of a shared factor scaling this source's rate; Poisson rates only. |
| `schedule` | the id of a schedule scaling this source's rate by the period's place in the cycle; Poisson rates only. |

### Nodes

A node is a state an event passes through. It can book values on
entry, book values over the time it holds the event, and choose where
the event goes next.

```yaml
- id: major_repair
  functions:
    - { name: parts,    category: materials,  basis: on_entry, logpert: [2e3, 8e3, 2e5] }
    - { name: downtime, category: production, basis: per_time, unit: days, pert: [1e3, 5e3, 2e5] }
  outcomes:
    - { to: replace,  p: 0.1, holding: { pert: [1, 3, 10], unit: days } }
    - { to: restored,         holding: { pert: [1, 3, 10], unit: days } }
```

| key | meaning |
|---|---|
| `id` | name, unique among nodes. |
| `description` | optional text. |
| `functions` | values booked at this node (below). |
| `outcomes` | where the event goes next, one of several (below). |
| `next` | a single next node with no selection; `next_holding` is the holding time before it, a distribution or `outage`. |
| `threshold` | where the event goes next by one of the node's values: above a level one way, at or below it the other (below). |
| `trigger` | the node is entered once at a period's end when a sum over the period is above a level, and by nothing else (below). |
| `mode` | `absolute` (default): each outcome's `p` is its share and at most one outcome may omit `p` to take the remainder. `sequential`: the outcomes are tried in order, each with its own probability, the last taking what is left. |

A node with no `outcomes`, `next` or `threshold` is a leaf, where the
event ends; a trigger's node is no different. The graph must be acyclic.

### Functions

| key | meaning |
|---|---|
| `name` | name, unique within the node. |
| `category` | the category the value is booked to; the report breaks totals down by category. |
| `basis` | `on_entry` (default): the value is booked once when the event enters. `per_time`: the value is a rate per time unit, booked over the holding time of the outcome the event takes; the node must have outcomes or a `next` with a holding. |
| `unit` | for `per_time`, the time unit of the rate. |
| `scope` | `event` (default): drawn afresh for every event. `period`: drawn once per period and shared by every event of that period. |
| `factor` | a factor's id (below): the value, or a `per_time` function's rate, is multiplied by the factor's draw for the period, shared with every value and rate naming it. |
| a distribution | the value, written inline beside the keys above: `pert: [...]`, `lognormal: {...}`, a number, and so on. |
| `expr` | in place of a distribution: the value as an expression of the node's other `on_entry` functions (below). |

#### Expressions

A function's value can be computed from the values its event draws at
the same node, in place of a distribution of its own:

```yaml
functions:
  - { name: hours, category: hours,  gamma: { mean: 8, sd: 2 } }
  - { name: wage,  category: wage,   gamma: { mean: 60, sd: 10 } }
  - { name: parts, category: parts,  lognormal: { median: 300, p90: 800 } }
  - { name: bill,  category: cost,   expr: "hours * wage + parts" }
outside_total: [hours, wage, parts]
```

An expression has numbers, the names of the node's `on_entry`
functions, `+`, `-`, `*` and parentheses; a name is that function's
value for the event, after any factor on it. It may name another
expression, except one with a factor, and not round a cycle. Each
function named is booked to its own category as usual, so a value read
only by an expression is usually kept outside the total.

Limits: 64 tokens, parentheses 16 deep, and 256 terms once the
expression is multiplied out. Once multiplied out no value may be
multiplied by itself, so `hours * hours` and `hours * (wage + hours)`
are refused; an expression that could be below 0, such as `parts - 500`,
is refused too. With those rules the exact mean and variance stay exact:
they need only each value's mean and variance.

An expression is booked on entry (`basis: on_entry`), drawn per event
(no `scope`) and has no range. It may not name a `per_time` function or
the value a threshold tests, be the value a threshold tests, or be a
trigger's input. A factor on an expression multiplies its value. The
exact distribution (the transform) is not computed for a model with an
expression; the mean, variance and simulation are.

### Outcomes

| key | meaning |
|---|---|
| `to` | the next node. |
| `p` | the probability: a number, or a distribution on [0, 1] (a `beta` for an uncertain share) drawn once per period. Omitted on at most one outcome, which takes the remainder. |
| `holding` | how long the event stays at this node before moving on: a distribution with a time `unit`, or the word `outage` for the time the reliability block that fired the event stays down (the node must be reached by exactly one source, a block with a repair). Values booked by `per_time` functions accrue over it, and a value may land in a later period than the event. |

#### The exact variance with holding times

A holding time lets a round book into later periods, so two of its
values land in one period only when they are less than a period apart.
For Poisson sources of fixed rates, with nothing drawn once a period,
the exact variance of the period's total is λ·E[Q], Q = ∫∫ (1 − |x −
y|)⁺ μ(dx) μ(dy), μ a round's bookings over the time from its arrival,
uniform in its period. It is computed by a backward pass over the nodes
on a grid of lags under one period against each holding's CDF, any
law's, the grid doubling until the figure settles to 1e-9; the
category, source, node and named total variances alike.

What is drawn once a period is shared by that period's rounds only,
and a period's total takes rounds from the periods before it, each with
its own draw. A rate drawn once a period (a mixed frequency, or a
factor on rates) adds Σ_{k,l} (E[Λ_k Λ_l] − λ_k λ_l)·Σ_d W_k(d)·W_l(d),
W_k(d) the mean a round of source k books into the period d after its
own; a value factor, a value or a probability drawn once a period adds
Σ_d Var(M(d)) of the period's mean booking by lag over the draw, values
and probabilities exactly on two points each; a holding drawn once a
period adds the covariance of the bookings it shifts over its law. The
mean bookings by lag are taken on a second grid over the periods the
holdings reach, to 1e-9 of the figure. A schedule's lags are not
independent draws but one sequence: the figure is the cycle's, the
schedule taken as a rate drawn each period from its values, with the
variance of the mean booked from all lags at each place of the joint
cycle in place of each lag's on its own. A path through two holdings
drawn once a period, holdings all but in lockstep, a schedule beside
a value factor or a value or probability drawn once a period, triggers,
repairs or queues still leave no exact variance with holding times, as
does a mixture of more than 64 points or one whose work is past its
cap.

### Thresholds

A threshold sends the event one way when a value the node draws is
above a level and the other way when it is at or below it: a repair
whose cost is above the insurance excess goes on to a claim, the rest
end there.

```yaml
- id: repair
  functions:
    - { name: callout, category: labour, pert: [500, 1200, 4000] }
    - { name: parts,   category: parts,  lognormal: { median: 8000, p90: 30000 } }
  threshold: { function: parts, over: 25000, above: claim, below: done }
```

| key | meaning |
|---|---|
| `function` | the name of the node's function whose value is tested; may be left out when the node has one function. |
| `over` | the level. Above is strictly greater, so a value exactly at it (a cap at the level, say) goes below. |
| `above`, `below` | the next node on each side. |

The value is drawn once and booked as usual, and the event branches on
it, so the branch and the value go together: the claims are exactly
the costly repairs. The exact mean, variance, P(T = 0) and distribution
split the value at the level into its two sides, P(value > level) with
the value given that it is above and the rest with the value given that
it is at or below, so they stay exact. A threshold is the node's
selection: the node has no `outcomes`, `next` or `mode`, its branches no
holding time and so no `per_time` function, and the tested value is
drawn per event, not with `scope: period`. The outcomes are reported as
`<node>→<above>` and `<node>→<below>`. See `examples/threshold.yaml`.

### Triggers

A trigger enters its node once, at the end of a period, when what the
period's events booked at some nodes, or how often they entered them,
adds up to more than a level: a year with more than three pump
failures sets off a fleet-wide inspection.

```yaml
- id: inspection
  trigger: { over: 3, entries: [failure] }
  functions:
    - { name: inspect, category: inspection, pert: [20000, 40000, 90000] }
  outcomes:
    - { to: redesign, p: 0.3 }
    - { to: clear }
```

| key | meaning |
|---|---|
| `over` | the level, finite and not negative. The trigger fires when the sum is strictly greater. |
| `entries` | the nodes whose entries it counts. The count books nothing, so the total stays in its own unit. |
| `inputs` | or the values it sums: a node for every function of it, or `node.function` for one. |

A trigger has `entries` or `inputs`, not both. It fires at most once a
period, after every source's events of the period, and its round
books in that period. A booking counts toward the period its event
arrived in, whatever period a holding time lands it in. No source or
outcome may lead into a trigger's node; otherwise the node is
ordinary: its functions book and its outcomes walk on. Several
triggers on the same inputs with different levels give a tiered
response, each firing once.

Triggers chain: an input may be another trigger's node, or a node its
walk reaches, and that trigger then fires first in the period, so its
bookings count. A trigger whose own firing feeds its sum, directly or
through others, is refused with the cycle named. Counting the entries
of trigger nodes gives a fault tree's gates per period: OR is a top
trigger over 0 on the basic events' marks, AND over n − 1, k out of n
over k − 1, each mark a trigger over 0 on one basic event. The meaning
is "happened in the same period", not "down at the same time": exact
for a mission of one period without repair; coincident down time
belongs to a reliability block. The top trigger's walk is a bow-tie's
consequence side. See `examples/bowtie.yaml`.

The report counts a trigger as a source named after its node: its
events are the firings, and its rows break down what the walk books. The
sum it tests is a quantity of its own, `trigger <node>`, with the
statistics, histogram and series column of any quantity, and the
analytic report gives its exact mean beside P(fire). The share of
periods in which it is above a level c is P(fire) at that level, so in
a capacity model one run gives the whole P(breach)-against-capacity
curve from that quantity: `variate run --exceedance curve.csv
--exceedance-of trigger:<node>` writes its curve P(S ≥ c), which for a
sum of continuous values is P(S > c), and for a count of entries is
P(fire) at c − 1. The quantity's quantiles answer the other way round:
its p95 is the level that fires one period in twenty. A level that
varies with a schedule, a crew on leave in August, is read the same
way one phase at a time: `--phase i` runs that place in the cycle, and
its curve read at that phase's level is the phase's P(fire). See
`examples/capacity.yaml`.
P(fire) is exact, a bracket within `transform_tol` from the transform
of the period's sum at the level,
and the means its walk books are exact with it. A chained trigger that
counts other triggers' firings is exact when those share nothing (no
source, shared factor, schedule or period-scoped value): the count is
then a sum of independent indicators, so OR, AND and k out of n are
exact. Marks that share a source (a repeated event in a fault tree), or
a chained trigger on booked values, leave P(fire) to the run. Where that transform
does not apply (a source with repair or a reliability block, a sum of
`per_time` values with holding times), P(fire) and those means are
unknown and the run measures them. The variance, the breakdowns'
variances and the total's transform are unknown with a trigger, since
the firing moves with the rest of the total, with one exception: a
model's only trigger whose `inputs` are every value the sources book
(a budget on the whole spend), in a model without holding times. The
total is then the spend plus the trigger's walk when the spend is over
the level, and its variance (within the transform's tolerance) and
exact distribution follow from the spend's transform, split at the
level. See `examples/budget.yaml`. P(T = 0) stays exact,
unless a trigger counts entries into a node that may book nothing,
which can fire in a period whose events booked nothing. The
plan counts the firings toward the minimum hits. See
`examples/trigger.yaml`.

## Distributions

A distribution is a bare number, meaning a constant, or a map with
exactly one family key. Beside the family key may stand `unit` (time
unit of a holding, a per-time rate or a frequency), `t` (the shape of a
PERT, default 4), `truncate: [lo, hi]`, `cap: c`, `scope` and `range`
(how far the input could be off, for `variate sweep`; below). A
constant that takes a range is written `{ constant: 0.3, range: [0.2,
0.4] }`.

| family | written as | notes |
|---|---|---|
| constant | `c` or `{ constant: c }` | |
| uniform | `{ uniform: [a, b] }` | |
| log-uniform | `{ loguniform: [a, b] }` | `a > 0` |
| triangular | `{ triangular: [low, mode, high] }` | |
| PERT | `{ pert: [low, peak, high] }`, optional `t` | the usual three-point elicitation |
| log-PERT | `{ logpert: [low, peak, high] }` | PERT on the logarithm: for values spanning decades |
| normal | `{ normal: { mean, sd } }` | needs `truncate` to enter a model, since values must be bounded below |
| truncated normal | `{ truncnormal: { mean, sd, lo, hi } }` | `hi` may be `.inf` |
| lognormal | `{ lognormal: { mu, sigma } }`, `{ median, p90 }` or `{ mean, sd }` | |
| exponential | `{ exponential: { mean } }` or `{ rate }` | |
| gamma | `{ gamma: { shape, scale } }`, `{ shape, rate }` or `{ mean, sd }` | as a frequency, gives negative-binomial counts |
| beta | `{ beta: [alpha, beta] }` or `{ beta: { mean, sd } }` | the natural uncertain probability |
| Weibull | `{ weibull: { shape, scale } }` | |
| Pareto | `{ pareto: { min, alpha } }` | mean finite only for `alpha > 1` |
| log-logistic | `{ loglogistic: { shape, scale } }` | `scale` is the median; mean finite only for `shape > 1`, variance only for `shape > 2`; as a holding time, the heavier its tail the longer the burn-in, and a tail that would need more than 2³¹ periods of it is refused: give that one a `truncate` |
| generalised Pareto | `{ gpd: { loc, scale, shape } }` | |
| Bernoulli | `{ bernoulli: p }` | |
| binomial | `{ binomial: { n, p } }` | |
| Poisson | `{ poisson: lambda }` | |
| negative binomial | `{ negbin: { r, p } }` or `{ mean, size }` | |
| geometric | `{ geometric: p }` | failures before the first success |
| categorical | `{ categorical: { values: [...], p: [...] } }` | `p` sums to 1 |
| empirical | `{ empirical: [x1, x2, ...] }` or `{ empirical: { file: data.csv } }` | resampling; the file sits beside the model |

`truncate: [lo, hi]` keeps draws inside the bounds by rescaling the
distribution to them. `cap: c` clamps draws at `c`, which puts an atom
of probability at the cap, as an insurance limit does. Values, holding
times and rates must not be negative; a `normal` needs a truncation at
zero or above.

A distribution whose mean is infinite (a Pareto with `alpha` at most
1) is refused. One with a finite mean and infinite variance (a Pareto
with `alpha` at most 2, a generalised Pareto with `shape` at least ½)
is allowed with a warning, as a value, a per-time rate or a frequency:
the standard errors of what it books into or reaches are marked
unreliable (`stderr_unreliable` in the JSON, a `stderr unreliable` row
in the CSV), and the `target_rse` and `contrib_rse` criteria on them
read "not applicable: infinite variance" and do not hold the run. The
quantiles and their errors stay valid. A `cap` or `truncate` gives a
finite variance. A per-time rate over a long holding time adds no
infinite variance, as a round books at most its rate in a period.

## Shared factors

Sources are independent unless they share a factor. A factor is drawn
once per period and multiplies the rate of every source naming it, so
a bad year is bad for all of them at once: the common-shock model,
which makes the aggregate tail heavier than independence says. The
dependence is exact in the variance, in P(T = 0) and in the exact
distribution, which mixes over the factor as over any per-period
quantity.

```yaml
factors:
  - { id: climate, lognormal: { mean: 1, sd: 0.48 } }
graphs:
  - id: register
    sources:
      - { id: phishing, frequency: { pert: [5, 12, 30] }, factor: climate, to: phishing_loss }
      - { id: malware,  frequency: { pert: [1, 3, 8] },   factor: climate, to: malware_loss }
```

A factor with mean 1 leaves every mean unchanged and adds only the
dependence; a factor's mean scales the rates it touches. Write a
lognormal factor by its mean and sd for that reason: `{ median: 1,
p90: 1.8 }` has mean 1.11, so it raises every rate it touches by 11%.

A function can name a factor too, and its value is then multiplied by
the factor's draw: one shock to the size of values, as inflation or a
price index is, rather than to how often they happen. Every value and
rate naming the factor moves with it in the same period, so the values
of different nodes move together, and a factor on both a source's rate
and its values makes a bad period both busier and dearer.

```yaml
factors:
  - { id: prices, lognormal: { mean: 1, sd: 0.2 } }
graphs:
  - id: line
    nodes:
      - id: repair
        functions:
          - { name: spares, category: parts, lognormal: { median: 3000, p90: 12000 }, factor: prices }
```

The value multiplied is the value: a threshold tests it, a trigger sums
it and a layer pays on it. A constant factor is folded into the laws, so
a factor of 1 changes nothing. The exact mean and variance condition on
the factors: given their draws the model is the same model with those
values and rates scaled, and over a Gauss rule for each factor the
figures are mixed, the variance as the mean of the variances given the
draws plus the variance of the means. The moments given the draws are
polynomials in them, which a three-node rule integrates exactly; a
threshold or a trigger on a scaled value is not, and the rule doubles
until every figure settles, a figure that does not being simulated with
its error. With holding times and one value factor, the variance with
holding times is taken at three of its nodes and at a fourth, which
tests the quadratic through the three; where it holds, to 1e-9, the
other nodes read it, and otherwise each is taken. A factor on values
needs a finite variance for these figures. The exact distribution is not
computed for a model with a factor on a value.

## Values that move together

The values one round draws are independent unless a dependence group
ties them: a bad breakdown needs more parts and keeps the line down
longer.

```yaml
dependence:
  - { functions: [repair.spares, repair->inspect, inspect.report], rank: 0.6 }
```

A member is `node.function`, an `on_entry` value or a `per_time` rate,
or `node->to`, the holding time from `node` to `to` (its `next_holding`
or the holding of its outcome to `to`). The members may be at one node
or along a path. `rank` is the rank correlation of every pair, from 0,
independent, to 1, in lockstep; each member keeps its own distribution.
Each round draws one standard normal W for the group, the first time it
draws a member, and each member is its law's quantile at
Φ(√ρ·W + √(1−ρ)·E), E its own normal and ρ = 2·sin(π·rank/6) the
normal correlation that gives the rank correlation. Rounds stay
independent of each other.

A member is drawn by its quantile rather than in batches, which costs
about 1 µs a draw for a gamma or PERT and 60 ns for a lognormal, against
10 to 30 ns; only members pay it. A group of rank 0 is none.

Every mean is unchanged: a node's rate and its holding may not both be
members, so no booking is the product of two values that move together.
The exact variance adds each pair's covariance, by Gauss–Hermite
quadrature, times the chance a round books both. With holding times
(above) each round is conditioned on its groups' normals, the members
then independent with their conditional laws, and the variance is the
Gauss–Hermite mixture over the normals, for values, rates and holdings
alike. Where a member holding moves a booking across a period's edge,
the figures bend in the normal, and where the Gauss–Hermite mixture
does not settle the normal is split at each bend and where its tails
start, with the normal's own Gauss rule on each piece, on a grid fine
enough for the holdings' spread given it. In lockstep (rank 1) a
holding given the normal is a point, spread over one cell of each grid
so that the figures do not bend at every cell, the spread's error going
with the grid's. So near lockstep that that grid is past its cap, as at
rank 0.99 over days, the variance with holding times is not computed.
P(T = 0) is not given when a member value can be 0.

A member may not be drawn once a period, be the value a threshold tests,
an expression or an expression's operand, be at a node a trigger reads,
at a queue node or after one, or be an outage holding. The exact
distribution (the transform) is not computed for a model with a group.

## Rates that vary by period

A schedule is a cycle of multipliers: period p uses the value at p
modulo the cycle's length, so with a monthly period twelve values are
a year and with a daily period seven are a week. A source naming a
schedule has its rate multiplied by the period's value. The series
shows the cycle; the exact mean, variance, P(T = 0) and distribution
treat the schedule as a multiplier seen equally often at each of its
values, which is what a run over whole cycles sees, so they describe a
period taken at random from the cycle. Every schedule steps with the
same period, so the schedules move together: those figures take them
jointly over their combined cycle, and two schedules peaking in the
same season raise the variance where opposite seasons lower it.

```yaml
period: { days: 30 }
schedules:
  - { id: storms, values: [0.3, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 2.2, 1.6, 0.8, 0.4, 0.3] }
graphs:
  - id: claims
    sources:
      - { id: storm, frequency: 4, schedule: storms, to: storm_claim }
```

A schedule's values are finite and not negative; a cycle of up to
65536 values. A schedule with mean 1 leaves the mean rate unchanged.

A rate that trends over a life, rather than repeating, is a schedule
over the life with a period of a year. Equipment that ages under minimal
repair (as old after a repair as before) fails as a Poisson process
whose rate rises with age; with the power-law (Weibull, Crow-AMSAA)
intensity of shape k the expected failures by age t are (t/η)^k, so over
a life of L years the value for year i, scaled to mean 1, is
L·(i^k − (i−1)^k)/L^k, and the source's frequency is the life's average
rate. The counts in each year are then exactly Poisson with that year's
mean. The whole cycle is a year drawn at random over the life and phase
i is year i. Equipment that is as good as new after each failure is a
renewal process instead: `every: n` with frequency n over the mean life
gives gamma lifetimes with a coefficient of variation of 1/√n. See
`examples/ageing.yaml`.

To ask about one place in the cycle instead, a January rather than a
month at random, give the top-level `phase: i` (or `--phase i` on the
command line, which overrides it): every schedule is fixed at its value
for that place, counted from 1, so every period of the run is that
phase and the exact figures and the distribution are those of one such
period. With several schedules the cycle is the least common multiple
of their lengths, and phase i takes each schedule's value for the
periods at that place: schedules of 4 and 6 values make a cycle of 12,
whose phase 8 takes the fourth value of the first and the second of the
other. A phase outside 1 to the cycle's length is refused with the
length, as is a phase of a model without schedules; 0, the default, is
the whole cycle. Every report of a phased model starts with
`phase i of n`.

## What the inputs could be: ranges

A distribution says how a value varies from event to event; the
simulation already covers that. A range says something else: how far
the distribution itself could be wrong, the uncertainty in the
estimate. `variate sweep` moves each input that has a range to the low
and to the high end of it, every other input as written, and ranks the
inputs by how far that moves a figure (the mean by default; `--figure`
takes a quantile, a return period, P(T = 0) or a layer's, category's
or source's mean). Inputs without a range are not moved.

```yaml
sources:
  - { id: wear_trip, frequency: { pert: [20, 60, 200], range: [50, 120] }, to: trip }   # 50 to 120 a year
  - { id: grid_outage, frequency: { pert: [0, 0.3, 2], range: { times: [0.5, 3] } }, to: fault_check }
nodes:
  - id: fault_check
    outcomes:
      - { to: major_repair, p: { constant: 0.75, range: [0.6, 0.85] },
          holding: { lognormal: { median: 2, p90: 7 }, unit: days, range: [2, 6] } }
```

`range: [low, high]` is the input's **mean** at the two ends, in the
distribution's own unit (a holding in days takes its range in days, a
rate per day its range per day) and a probability as probabilities.
The whole distribution is scaled to reach each end, so its shape and
spread stay; the range must hold the input's mean. `range: { times:
[low, high] }` gives the factors instead, which suits an input without
a natural unit, and is the form to use on a demo knob, whose mean
moves. A schedule takes `range` beside its `values`, for the mean of
the cycle; a factor beside its family; a block's `mtbf` or `mttr` as
`{ constant: 6000, range: [4000, 9000] }`, in the block's unit, named in
the sweep `source <id> block <n> mtbf` with `n` counting the block's
components and k-of-n groups from 1 in the order written. A k-of-n
group's units move together. A block whose components all lack a
repair time is a Poisson source at its rate, so its times take no
range: give the source a frequency with one instead. A probability moved past 1 is
held there, and a node's other probabilities make room so that it
still sums to 1.

Without ranges, `variate sweep --screen 0.1` moves every input by a
tenth either way. That ranks leverage, not uncertainty: it shows which
inputs are worth giving a range.

`variate compare -f model.yaml --option a.yaml --option b.yaml` sets
whole options against the model instead: each option is a model file,
changed in any way, an input moved, a node or source added, and run
with the model's settings, not its own. Each row is the option's
figure, the model's, and their difference, by the cheapest method
that gives both: exact, a bracket, or simulated at the model's seed
for its run's length, with the difference's standard error measured
on paired blocks. Draws are keyed by what they are, so an option draws
as the model does wherever it does not differ, and that error is
small.

## Queues

A node with a `queue` serves the rounds that reach it in order of
arrival, from every source, with limited capacity: what cannot be
served at once waits.

```yaml
- id: backlog
  queue:
    servers: 2                 # default 1
    speed: 1                   # work each server does per period; default 1
    work: { lognormal: { median: 0.5, p90: 3 } }   # each round's work, drawn on arrival
    period_end: carry          # carry the line on, or reset it at each period's end
  functions:
    - { name: open, category: exposure, basis: per_time, constant: 400 }
  outcomes:
    - { to: closed }
```

A round reaching the node draws its work, which may take a time `unit`
as a holding time does, and joins the line; a free server takes the
round that has waited longest and serves it for work ÷ speed periods.
The node's on-entry functions book when the round arrives, its
per-time functions over its stay, waiting and service, and on being
served the round goes on by the node's outcomes, holding times
included, which the per-time functions book over too, as at any node.

`period_end: reset` empties the line at each period's end: the rounds
still waiting or in service go to the node named by `unfinished`,
which books and routes them like any other. `carry` keeps the line
from one period to the next, which needs a started run (below): in
steady state its warm-up has no known bound. In a steady-state run a
reset queue must be reached within the period its rounds arrive, so no
holding time, trigger or `unfinished` exit may lie on a path into it.

The report gives, per queue, the wait and the stay of the rounds it
served, each with its mean, quantiles and their errors, which come
from groups of tasks since rounds that share a line are not
independent; and per period (per repeat in a started run) the rounds
that joined, those served and those sent to `unfinished`, the mean
line at a period's end and the share of the servers' time busy. A
started run also gives, per period, the mean wait and stay of the
rounds that arrived in it, with their errors across the repeats, and
the line and busy time at each period's end. When the work, or the
count of a source reaching a carried queue, has infinite variance, so
do the waits and stays, and their standard errors are marked
unreliable; a reset queue's are bounded by the period.

A carried queue whose rounds arrive as a Poisson stream at a constant
rate also gets its exact long-run figures, the stationary wait and
stay it settles to once it has run a long time, which the late periods
of a long horizon approach: by Erlang C for exponential work at any
number of servers, and by Pollaczek–Khinchine for one server and any
work. They are the mean wait and stay, and the wait's quantiles, exact
under Erlang C and under Pollaczek–Khinchine each an interval that
holds it, computed from bounds on the law. The input is Poisson when
every source reaching the queue has a constant frequency, with no
factor, schedule, `every`, reliability block, repair time or trigger,
and the paths into it pass no other queue and nothing drawn once a
period; holding times and branching keep it Poisson. With a load of 1
or more, several servers of other work, or other input, the report
says why there are none. What a queue books after
it, and over its stay, has no exact mean; what is booked before it
keeps one. A trigger whose inputs are reached through a queue, an
outage holding after one, and the round stream with queues are not
supported yet.

## Started runs

A run is otherwise the steady state: every period alike, as if the
model had always been running, which a burn-in before period 0 makes
true. A started run instead begins at period 0 with nothing in
progress, and asks what happens in period 1, 2, … of a horizon:

```yaml
start: { from: empty, horizon: 12 }   # twelve periods from an empty start
repeats: 1000                         # optional; omitted, the plan decides
```

Each repeat simulates the horizon once from the start, from its own
random streams. Nothing is in flight at period 0: no round is in a
holding time, and nothing booked before 0 counts. The sources start in
their steady state, as always: a reliability block's units, a repair
source's silence and an `every` source's count are drawn from their
long-run law. What a repeat books past the horizon is dropped, and its
total is the report's `beyond horizon`.

The report then has two parts. The usual figures describe the
horizon: each repeat adds up its periods into one horizon total, and
the mean, quantiles, exceedance curve, layers (an annual
deductible applies to the horizon's total) and the breakdown are over
those totals, one per repeat; the criteria bind them and the plan
counts repeats. A table by period gives, for each period of the
horizon, the mean across the repeats with its standard error and
standard deviation, and the exact mean beside it where the model gives
one; the JSON (`by_period`) and CSV (rows of quantity `period`) carry
every quantity this way: total, categories, sources, entries and value
per node, and each trigger's sum.

The exact mean of a period comes from the arrivals of every earlier
period and the share of each that has passed its holding time by then.
It applies where every path into a node passes at most one holding
time drawn from a law; a node behind two holdings in a row, or an
`outage` holding, and a trigger's rounds, are simulated with their
error. A period's rate follows its schedule from the cycle's first
value, so `phase` is refused: to start in another season, write the
schedule's values from that season. The exact variance, P(T = 0) and
the exact distribution assume stationary periods and are not computed,
so `transform` is refused, as are `burn_in` and `block_periods`, since
a repeat is one task. `--series` keeps every repeat's periods, one
after another, and the round stream gives each round its repeat.
`variate sweep` and `variate compare` give the figures a started run
has exactly, the horizon's mean and its categories' and sources'
means.

### From a given state

A started run can instead begin from a system already running: the
rounds waiting or in service at each queue, and the rounds part-way
through a holding time, each said with a count and distributions, and
observed data with `empirical: { file }`:

```yaml
start:
  from: given
  horizon: 24
  state:
    - { node: backlog, count: 30, age: { empirical: { file: data/open_ages.csv } }, in_service: { constant: 1 } }
    - { node: inspection, to: closed, count: { poisson: 4 } }
```

- An entry without `to` puts `count` rounds in the queue `node`, each
  there for a time drawn from `age`. The ages are sorted: the oldest,
  as many as the queue has servers, are in service, and the rest wait
  in line, oldest first. A waiting round's work is drawn as an
  arrival's. A round in service has served for `in_service`, and its
  remaining work is the work's law given that it is more than that
  time's work; without `in_service`, its remaining work and the time it
  has served come from the work's long-run (equilibrium) residual, as
  in a queue that has long been running. A round's wait and stay count
  its age, and it counts as joining in period 0.
- An entry with `to` puts `count` rounds part-way through the holding
  time of `node`'s outcome (or `next`) to `to`, each held for a time
  drawn from `elapsed`. What is left of the holding is drawn from its
  law given that it has lasted that long, past the law's end nothing;
  without `elapsed`, from its equilibrium residual. The rounds then go
  on to `to`.
- The equilibrium residual of a law with infinite variance (a Pareto
  with `alpha` at most 2) has an infinite mean, so such an entry must
  give `in_service` or `elapsed`.
- `source` names whose rounds they are, in the breakdown; it may be
  left out when the model has one source. `count` is a number or a
  distribution drawn once per repeat, rounded to a whole number.
- Nothing booked before period 0 is booked again: no on-entry value at
  `node`. The per-time values at `node` book from 0 over what is left
  of the stay or the holding.
- A round's age and its time in service are drawn independently, so a
  round can have served longer than it has been there; its wait then
  counts as 0.

The time units are the model's, and `unit:` converts as for a holding
time. The report says which entries drew what was left from the
equilibrium. An `outage` holding is refused, as its length is that of
the block's time down, which a state does not hold. What the given
rounds reach has no exact per-period mean and is simulated; another
source's rounds keep theirs. The round stream is refused, as it follows
a round from its arrival.

### Discounting

A started run can give present values: a value booked in a later period
of the horizon is worth less today than the same value in period 1.
Discounting is off unless the model asks for it, and applies only to
the categories it names:

```yaml
discount: { rate: 0.05, categories: [revenue, labour, parts] }
```

A value in a named category booked in period p is booked times
(1 + rate)^-p, discounted to the start from the end of its period;
`factors: [...]`, one per period of the horizon, each 0 or more, give
any other curve in place of `rate`, such as (1 + rate)^-(p - 1/2) for
the middle of the period. A category not named, such as hours, is never
discounted.

A value is discounted once, as it is booked, and nowhere else. Every
figure of a named category is of present values: the table by period,
the horizon's total, its quantiles, layers, `--at`, `--when`, the
breakdowns by category, source and node, and a named total, which is
discounted through its categories and is refused in the list itself.
What the model does with a value is not discounted: what a node feeds a
trigger or a threshold, which rounds take which outcome and how long
they hold, so a discounted run simulates the same rounds as one without
it at the same seed. Each period's exact mean is the factor times the
undiscounted one. What a repeat books past the horizon, the report's
`beyond horizon`, is not discounted. A discount whose every factor is 1
changes nothing. The report names the categories and the first and last
factors, and the JSON gives `discount` with every factor.

## Layers of the period total

A layer pays, in each period, the share of the total between its
attachment and the attachment plus its limit: `share × min(max(total −
attachment, 0), limit)`. `attachment` defaults to 0, `limit` to none and
`share` to the whole band. Layers are computed from each period's
total, so each has the statistics, histogram, exceedance curve and
series column of any quantity, and when the exact distribution applies
its mean comes with exact bounds.

```yaml
layers:
  - { id: retained, limit: 2e5 }                                      # under the aggregate deductible
  - { id: reinsurance, attachment: 2e5, limit: 3e6, share: 0.9 }      # the layer above it, 90% placed
```

## Totals the model names

Every value is booked to a category, and the period total is the sum
of the categories. A model can name totals of its own, each the sum of
the categories it lists, and keep categories outside the total:

```yaml
totals:
  - { id: maintenance, categories: [labour, parts] }               # the bill for a stoppage
  - { id: cost, categories: [labour, parts, production] }          # the bill and the lost output
  - { id: worked, categories: [hours] }                            # the crew's hours
outside_total: [hours]
```

A category in `outside_total` is booked and reported as any other, in
its own unit, but is added to neither the total nor the sources' and
nodes' values, which stay in the total's unit; it has no share of the
total, and it reaches a total only through a named one. The total is
then the sum of every other category, which a model naming no totals
and keeping no category outside has as before.

A named total is a quantity like the total: it has the statistics,
histogram, exceedance curve, series column and table by period of any
quantity, and is named `total:<id>` in `--at` and `--when`, as in
`--when total:maintenance>=1/100`. A condition lists the named totals
among its parts. Its exact mean is the sum of its categories', and its
exact variance and exact distribution are computed as the total's are,
over its categories' values; its share of the total is given when all
its categories are in it. Layers apply to the total alone.

A named total may also subtract categories, listed in `less`, as a net
revenue is the revenue less the cost:

```yaml
totals:
  - { id: net, categories: [revenue], less: [labour, parts, production] }
outside_total: [revenue]
```

Values themselves are never negative, but such a total can be, and
it is reported as any named total, with these differences: its
histogram counts negative values too, so its quantiles run below 0;
the report gives P(Q ≤ 0), the chance of a period at or below 0,
counted in the run with its standard error; it has no share of the
total; and its exact mean and variance are computed, but not its
exact distribution. Its exceedance curve covers its positive values.

## Reliability blocks

A `block` describes a system of components with exponential failure
and repair times. The engine simulates the components' failures and
repairs and fires the source each time the system goes down, so a
redundant block's failures cluster after a component failure as they do
in life. `variate plan` prints the block's stationary failure rate,
system MTBF and unavailability.

```yaml
block:
  unit: hours                 # the time unit of every mtbf and mttr below, unless an inner block says otherwise
  series:
    - { kofn: { k: 2, n: 3, mtbf: 6000, mttr: 36 } }    # three chillers, two needed
    - { kofn: { k: 1, n: 2, mtbf: 12000, mttr: 24 } }   # two pumps, one needed
    - { mtbf: 30000, mttr: 8 }                          # the control system
```

| form | meaning |
|---|---|
| `{ mtbf: m, mttr: r }` | one component: mean time between failures (its mean up time) and mean repair time. `mttr` omitted means replaced at once. |
| `{ constant: c, range: ... }` | in place of a number for `mtbf` or `mttr`: the time with a range for `variate sweep`, in the block's unit, as below. |
| `kofn: { k, n, mtbf, mttr }` | `n` identical components of which `k` must be working. |
| `series: [ ... ]` | blocks that must all be up; may nest. |

When no component has a repair time the block is a Poisson source at
its stationary rate. `period` must be declared. An outcome of the node
the block fires into may hold for `outage`, the time the block's own
repairs take to bring the system back, so a per-time cost over the
outage follows the repair that caused it.

## Templates

A template is a sub-graph instanced several times with different
names and wiring, for repeated structure such as one repair path per
machine.

```yaml
templates:
  - id: repair_path
    entry: assess                               # the node instances are entered at
    exits: [replace, restored]                  # nodes every instance must wire to
    nodes:
      - { id: assess, outcomes: [ { to: replace, p: 0.2 }, { to: restored } ] }
graphs:
  - id: site
    sources:
      - { id: pump_a, frequency: { pert: [1, 3, 10] }, to: pump_a_repair.assess }
    instances:
      - { template: repair_path, as: pump_a_repair, wire: { replace: new_pump, restored: recommission } }
      - { template: repair_path, as: pump_b_repair, wire: { replace: new_pump, restored: recommission },
          override: { "assess.outcomes[0].p": 0.35 } }
    nodes:
      - { id: new_pump, functions: [ { name: new_pump, category: capital, logpert: [5e4, 1e5, 1e6] } ] }
      - { id: recommission }
```

An instance's nodes are named `<as>.<node>`; `wire` maps each exit to
a node of the graph; `override` replaces a value at a path inside the
instance.

## Settings

Any of these may appear at the top level of the file, and each has a
command-line flag of the same name with hyphens (`--min-hits`). The
flag wins.

| setting | meaning |
|---|---|
| `periods` | run exactly this many periods. Omitted: the plan decides, running until every criterion below is met. |
| `repeats` | in a started run, run exactly this many repeats of the horizon; omitted, the plan decides. A started run refuses `periods`. |
| `seed` | the random seed; 0, the default, draws a fresh seed each run, which the report gives. The same seed, model and `block_periods` repeat a run exactly. Every draw is keyed by what it is: the period, the source, the round's number among the source's rounds that period, and the node, function, holding or choice by name. Two models at one seed so draw alike wherever an edit does not reach: a changed probability moves only the rounds between its old and new thresholds, a changed law redraws only its own values, and an added node, function or source leaves every other draw as it was. Their periods pair, and a difference between them is measured far more sharply than from two independent runs. |
| `workers` | simulation threads; default all cores, or as many as fit `max_memory`. |
| `min_hits` | events every reachable outcome must see (default 1000). |
| `contrib_rse`, `target_rse` | relative standard error asked of each node's contribution and of each mean (default 1%). |
| `tail`, `tail_hits` | the tail quantile to resolve (default 0.999) and the periods wanted beyond it (default 100). |
| `max_periods`, `max_steps` | hard caps on an adaptive run. |
| `allow_unconverged` | run to the caps when the plan exceeds them, instead of refusing. |
| `transform` | also compute the exact distribution by FFT at compile, where the model allows it; `transform_grid`, `transform_tol`, `transform_work` and `transform_bytes` set its largest grid, tolerance, work budget and the memory a pass may allocate (in the browser, unless set, a tolerance of 0.01 and 640 MB). The tolerance is met at a point when either P(T ≥ x) or the quantiles there are bracketed within it, relatively. |
| `block_periods` | periods per task; the default is 4096, or a thirty-second of `periods` in a fixed run. |
| `batch`, `burn_in`, `max_memory`, `round_scratch`, `record_buffer` | sizing of the run's buffers; the defaults suit nearly every model. |
| `hist_res` | the relative width of the histogram bins the quantiles come from (default 0.001). |
| `series` | keep every period's totals for output. |
| `no_breakdown` | skip the per-category, per-source and per-node contribution figures and their precision criteria, so the run is sized by the total alone: much faster when a breakdown sets the length. |
| `node_hist` | keep histograms of entries per node. |
| `pace` | when streaming rounds, play them at this duration per period (`1.5s`). |

## Worked examples

The `examples/` directory holds one model per feature, each run by the
test suite, and `examples/README.md` says what each shows.
