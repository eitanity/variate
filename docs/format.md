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
| `period` | the length of one period, as one of `{ hours: n }`, `{ days: n }`, `{ weeks: n }`. Needed when any distribution, holding time or block uses a time `unit`. |
| `graphs` | a list of graphs; sources and nodes of every graph form one model. |
| `templates` | reusable sub-graphs instanced inside graphs (below). |
| `factors` | shared multipliers on rates, each `{ id, <distribution> }` on [0, ∞), drawn once per period and shared by every source naming it: a common shock (below). |
| `schedules` | cycles of multipliers on rates, each `{ id, values: [...] }`; period p uses the value at p modulo the cycle's length: a season or a shift pattern (below). |
| `layers` | functions of the period total reported beside it, each `{ id, attachment, limit, share }`: an aggregate deductible, an aggregate limit or a reinsurance layer (below). |
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
| `mode` | `absolute` (default): each outcome's `p` is its share and at most one outcome may omit `p` to take the remainder. `sequential`: the outcomes are tried in order, each with its own probability, the last taking what is left. |

A node with no `outcomes` and no `next` is a leaf, where the event
ends. The graph must be acyclic.

### Functions

| key | meaning |
|---|---|
| `name` | name, unique within the node. |
| `category` | the category the value is booked to; the report breaks totals down by category. |
| `basis` | `on_entry` (default): the value is booked once when the event enters. `per_time`: the value is a rate per time unit, booked over the holding time of the outcome the event takes; the node must have outcomes or a `next` with a holding. |
| `unit` | for `per_time`, the time unit of the rate. |
| `scope` | `event` (default): drawn afresh for every event. `period`: drawn once per period and shared by every event of that period. |
| a distribution | the value, written inline beside the keys above: `pert: [...]`, `lognormal: {...}`, a number, and so on. |

### Outcomes

| key | meaning |
|---|---|
| `to` | the next node. |
| `p` | the probability: a number, or a distribution on [0, 1] (a `beta` for an uncertain share) drawn once per period. Omitted on at most one outcome, which takes the remainder. |
| `holding` | how long the event stays at this node before moving on: a distribution with a time `unit`, or the word `outage` for the time the reliability block that fired the event stays down (the node must be reached by exactly one source, a block with a repair). Values booked by `per_time` functions accrue over it, and a value may land in a later period than the event. |

## Distributions

A distribution is a bare number, meaning a constant, or a map with
exactly one family key. Beside the family key may stand `unit` (time
unit of a holding, a per-time rate or a frequency), `t` (the shape of a
PERT, default 4), `truncate: [lo, hi]`, `cap: c` and `scope`.

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
  - { id: climate, lognormal: { median: 1, p90: 1.8 } }
graphs:
  - id: register
    sources:
      - { id: phishing, frequency: { pert: [5, 12, 30] }, factor: climate, to: phishing_loss }
      - { id: malware,  frequency: { pert: [1, 3, 8] },   factor: climate, to: malware_loss }
```

A factor with mean 1 leaves every mean unchanged and adds only the
dependence; a factor's mean scales the rates it touches.

## Rates that vary by period

A schedule is a cycle of multipliers: period p uses the value at p
modulo the cycle's length, so with a monthly period twelve values are
a year and with a daily period seven are a week. A source naming a
schedule has its rate multiplied by the period's value. The series
shows the cycle; the exact mean, variance, P(T = 0) and distribution
treat the schedule as a multiplier seen equally often at each of its
values, which is what a run over whole cycles sees, so they describe a
period taken at random from the cycle. Two sources on the same
schedule rise and fall together, and the variance carries that.

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
| `seed` | the random seed; 0 means a fresh seed each run. The same seed, model and `block_periods` repeat a run exactly. |
| `workers` | simulation threads; default all cores. |
| `min_hits` | events every reachable outcome must see (default 1000). |
| `contrib_rse`, `target_rse` | relative standard error asked of each node's contribution and of each mean (default 1%). |
| `tail`, `tail_hits` | the tail quantile to resolve (default 0.999) and the periods wanted beyond it (default 100). |
| `max_periods`, `max_steps` | hard caps on an adaptive run. |
| `allow_unconverged` | run to the caps when the plan exceeds them, instead of refusing. |
| `transform` | also compute the exact distribution by FFT at compile, where the model allows it; `transform_grid`, `transform_tol`, `transform_work` and `transform_bytes` set its largest grid, tolerance, work budget and the memory a pass may hold. |
| `block_periods` | periods per task; the default is 4096, or a thirty-second of `periods` in a fixed run. |
| `batch`, `pool_size`, `spill`, `max_memory`, `round_scratch`, `record_buffer` | sizing of the run's buffers; the defaults suit nearly every model. |
| `hist_res` | the relative width of the histogram bins the quantiles come from (default 0.001). |
| `series` | keep every period's totals for output. |
| `no_breakdown` | skip the per-category and per-source breakdowns. |
| `node_hist` | keep histograms of entries per node. |
| `pace` | when streaming rounds, play them at this duration per period (`1.5s`). |

## Worked examples

The `examples/` directory holds one model per feature, each run by the
test suite, and `examples/README.md` says what each shows.
