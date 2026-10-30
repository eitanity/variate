# Variate

A Monte Carlo engine for event-driven risk and cost models, in Go,
with exact answers wherever the model allows them.

A model is a directed acyclic graph. Sources emit events at a rate per
period; each event walks the graph from the node it enters, booking
values at the nodes it passes, choosing an outcome at each, and holding
for a while where the model says so. The engine simulates periods and
reports the period total, its breakdown by category and source, the
events and values at each node, quantiles, the exceedance curve and
who the tail belongs to. Beside the simulation it computes the exact
mean, variance and probability of an empty period from the graph, and
for most models the exact distribution of the period total by FFT,
bracketed between rigorous bounds, with no sampling at all.

## What a model can say

- **Rates:** a constant or an uncertain rate drawn once per period
  (Poisson or mixed-Poisson counts); a source that fires on every n-th
  arrival of its stream; a source silent for a repair after each firing;
  a reliability block diagram of components with mean times between
  failures and repairs, simulated exactly as a Markov chain of the
  units; and shared factors that scale several rates at once, the
  common shock that makes a bad year bad for every risk together; and
  schedules, cycles of multipliers that give a rate its season or
  shift pattern.
- **Nodes:** values booked on entry or per unit of time held, in any
  of twenty-two distribution families with truncation and caps; a
  selection of outcomes with fixed or uncertain probabilities, as
  shares or tried in order; holding times, including the outage of the
  reliability block that fired the event; values and probabilities
  drawn once per period and shared by every event in it.
- **Structure:** several graphs in one model, templates instanced with
  overrides, categories added across the model.

`docs/format.md` is the reference for the YAML model file, and
`examples/` holds seventeen complete models with a README saying what
each shows, from a one-line compound loss to a register of a hundred
risks, a plant as a block diagram, and a shift with its own period.

## Running

    go run ./cmd/variate run -f examples/compound.yaml          # simulate to the plan's criteria
    go run ./cmd/variate run -f examples/compound.yaml --analytic   # with the exact distribution
    go run ./cmd/variate plan -f examples/compound.yaml         # the convergence plan and work
    go run ./cmd/variate stream -f examples/sequential.yaml --leaf undetected --rounds csv

`run` reports a table, JSON or CSV; `plan` says how many periods each
convergence criterion needs and what the run will cost; `stream`
writes the rounds, the path each event takes with the values it
books, filtered to a node or a leaf. A run without a period count
continues until every criterion is met: enough hits on every outcome,
a small enough error on every mean, enough periods in the tail. The
same model, seed and block size repeat a run exactly.

## The browser demo

`make demo` builds the engine to WebAssembly with TinyGo and serves a
page at http://127.0.0.1:8000/ with every example as a model with
knobs, or any model pasted as YAML. A run draws the histogram and
exceedance curve of the total or of any category, source or node, with
return periods and the exact bounds when asked for; the sources'
contributions and who the tail belongs to beyond a return period; a
source's events per period against a Poisson reference; the totals
over the run; and the compiled graph as a flow diagram. A run can be
pinned as a baseline and the next drawn beside it, for
cheaper-and-slower against faster-and-dearer. The page lives under
`web/`; `cmd/variate-wasm` is the engine's browser entry point.

## From Python

The `variate-python` repository, MIT-licensed, builds a model in
Python and runs it through the `variate` command, the model crossing as
a YAML file and the report returning as JSON. The engine runs as a
separate program, so a notebook, a library or a product built on the
package takes on none of the engine's obligations; embedding the engine
itself is what this repository's licence and its commercial licences
cover. `variate plan --format json` and the series in the JSON report
exist for it.

## Building and testing

`make ci` runs what runs on every push: formatting, vet, the linters
and the project's own analyzers, the escape and allocation checks on
the hot path, the race detector, the suite again with a 32-bit `int`
(the width of TinyGo's wasm target), a TinyGo build of the browser
engine, the fuzzers and the vulnerability scan. `make stat` is the
long statistical suite, `make bench` the benchmarks against a
baseline, `make wasm-test` the engine's tests as wasm under the wazero
runtime, and `make race` the race detector run. The engine's test
suites live in `internal/suite`, against the public API only; the root
keeps the tests that reach inside. `internal/lint` is a separate
module holding the project's checkers. The engine allocates nothing
while a run is going, has no recursion and bounds every loop, and the
build checks each of these with a tool wherever a tool can.

## Licence

Copyright © 2026 Eitanity Systems VCC.

Variate is free software under the GNU Affero General Public License,
version 3 (see `LICENSE`). Any work that links it, calls it over a
network or ships it must be offered under the same terms, source
included. Commercial licences are available for those terms do not
suit.
