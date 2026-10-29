package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

// builder validates a Graph and flattens it into a Compiled.
type builder struct {
	factorIndex   map[string]int32
	scheduleIndex map[string]int32
	g             Graph
	cfg           Config
	nodeIndex     map[string]int32
	warnings      []Warning
}

func (b *builder) validate() error {
	if err := b.checkIDs(); err != nil {
		return err
	}
	for _, n := range b.g.Nodes {
		if err := b.checkNode(n); err != nil {
			return err
		}
	}
	for _, s := range b.g.Sources {
		if err := b.checkSource(s); err != nil {
			return err
		}
	}
	if err := b.checkAcyclic(); err != nil {
		return err
	}
	if len(b.g.Sources) == 0 {
		return modelError("graph", b.g.ID, "no sources")
	}
	return b.checkWatch()
}

func (b *builder) checkIDs() error {
	for i, n := range b.g.Nodes {
		if n.ID == "" {
			return modelError("node", itoa(i), "empty id")
		}
		if _, dup := b.nodeIndex[n.ID]; dup {
			return modelError("node", n.ID, "duplicate id")
		}
		b.nodeIndex[n.ID] = index(i)
	}
	seen := map[string]bool{}
	for _, s := range b.g.Sources {
		if s.ID == "" || seen[s.ID] {
			return modelError("source", s.ID, "empty or duplicate id")
		}
		seen[s.ID] = true
	}
	if err := checkLayers(b.g.Layers); err != nil {
		return err
	}
	b.scheduleIndex = map[string]int32{}
	for i, sc := range b.g.Schedules {
		if err := checkSchedule(i, sc, b.scheduleIndex); err != nil {
			return err
		}
		b.scheduleIndex[sc.ID] = index(i)
	}
	b.factorIndex = map[string]int32{}
	for i, f := range b.g.Factors {
		if f.ID == "" {
			return modelError("factor", itoa(i), "empty id")
		}
		if _, dup := b.factorIndex[f.ID]; dup {
			return modelError("factor", f.ID, "duplicate id")
		}
		if err := checkSlot("factor", f.ID, f.Model, 0, math.Inf(1), false); err != nil {
			return err
		}
		b.factorIndex[f.ID] = index(i)
	}
	return nil
}

func (b *builder) checkSource(s Source) error {
	if _, ok := b.nodeIndex[s.To]; !ok {
		return modelError("source", s.ID, "target node "+s.To+" not found")
	}
	if s.Factor != "" {
		if _, ok := b.factorIndex[s.Factor]; !ok {
			return modelError("source", s.ID, "factor "+s.Factor+" not found")
		}
		if len(s.Chains) > 0 || s.Every > 1 || s.Repair != nil {
			return modelError("source", s.ID, "a factor scales a Poisson rate only")
		}
	}
	if s.Schedule != "" {
		if _, ok := b.scheduleIndex[s.Schedule]; !ok {
			return modelError("source", s.ID, "schedule "+s.Schedule+" not found")
		}
		if len(s.Chains) > 0 || s.Every > 1 || s.Repair != nil {
			return modelError("source", s.ID, "a schedule scales a Poisson rate only")
		}
	}
	if len(s.Chains) > 0 {
		return checkChains(s)
	}
	if err := checkSlot("source", s.ID, s.Frequency, 0, math.Inf(1), false); err != nil {
		return err
	}
	if s.Every < 0 || s.Every > maxEvery {
		return modelError("source", s.ID, "every must be between 0 and "+itoa(maxEvery))
	}
	if s.Repair == nil {
		return nil
	}
	if s.Every > 1 {
		return modelError("source", s.ID, "every and repair are exclusive")
	}
	if lo, hi := s.Frequency.Support(); lo != hi {
		return modelError("source", s.ID, "repair needs a constant frequency")
	}
	return checkSlot("source", s.ID+".repair", s.Repair, 0, math.Inf(1), false)
}

// checkSchedule validates a schedule: a unique id and finite values
// that are not negative, at least one of them.
func checkSchedule(i int, sc Schedule, seen map[string]int32) error {
	if sc.ID == "" {
		return modelError("schedule", itoa(i), "empty id")
	}
	if _, dup := seen[sc.ID]; dup {
		return modelError("schedule", sc.ID, "duplicate id")
	}
	if len(sc.Values) == 0 || len(sc.Values) > maxScheduleLength {
		return modelError("schedule", sc.ID, "needs 1 to "+itoa(maxScheduleLength)+" values")
	}
	for _, v := range sc.Values {
		if !(v >= 0) || math.IsInf(v, 1) {
			return modelError("schedule", sc.ID, "values must be finite and not negative")
		}
	}
	return nil
}

// checkLayers validates the layers: unique ids, a finite attachment,
// a positive limit or none, a share in (0, 1] or the whole band.
func checkLayers(layers []Layer) error {
	seen := map[string]bool{}
	for i, y := range layers {
		switch {
		case y.ID == "" || seen[y.ID]:
			return modelError("layer", itoa(i), "empty or duplicate id")
		case !(y.Attachment >= 0) || math.IsInf(y.Attachment, 1):
			return modelError("layer", y.ID, "attachment must be finite and not negative")
		case y.Limit < 0 || math.IsNaN(y.Limit):
			return modelError("layer", y.ID, "limit must be positive, or 0 for none")
		case y.Share < 0 || y.Share > 1 || math.IsNaN(y.Share):
			return modelError("layer", y.ID, "share must be in (0, 1], or 0 for the whole band")
		}
		seen[y.ID] = true
	}
	return nil
}

// checkChains validates a reliability diagram source: no other count
// law, and every chain well formed with finite rates.
func checkChains(s Source) error {
	if s.Frequency != nil || s.Every > 1 || s.Repair != nil {
		return modelError("source", s.ID, "chains stand in for frequency, every and repair")
	}
	for i, c := range s.Chains {
		switch {
		case c.N < 1 || c.N > maxChainUnits || c.K < 1 || c.K > c.N:
			return modelError("source", s.ID, "chain "+itoa(i)+": need 1 ≤ k ≤ n ≤ "+itoa(maxChainUnits))
		case !(c.Fail >= 0) || math.IsInf(c.Fail, 1) || !(c.Repair >= 0) || math.IsInf(c.Repair, 1):
			return modelError("source", s.ID, "chain "+itoa(i)+": rates must be finite and not negative")
		}
	}
	return nil
}

// checkOutage: an outage holding stands alone, without a distribution.
func checkOutage(n Node) error {
	if n.NextOutage && n.NextHolding != nil {
		return modelError("node", n.ID, "next_holding is outage or a distribution, not both")
	}
	for _, o := range n.Outcomes {
		if o.OutageHolding && o.Holding != nil {
			return modelError("outcome", n.ID+"→"+o.To, "holding is outage or a distribution, not both")
		}
	}
	return nil
}

func (b *builder) checkNode(n Node) error {
	if err := checkOutage(n); err != nil {
		return err
	}
	if len(n.Outcomes) > 0 && n.Next != "" {
		return modelError("node", n.ID, "has both outcomes and next")
	}
	leaf := len(n.Outcomes) == 0 && n.Next == ""
	for _, f := range n.Functions {
		if err := checkSlot("function", n.ID+"."+f.Name, f.Model, 0, math.Inf(1), false); err != nil {
			return err
		}
		if f.Basis == PerTime && leaf {
			return modelError("function", n.ID+"."+f.Name, "per-time function on a leaf: nothing holds it")
		}
	}
	if n.Next != "" {
		if _, ok := b.nodeIndex[n.Next]; !ok {
			return modelError("node", n.ID, "next node "+n.Next+" not found")
		}
		return checkHolding(n.ID+"→"+n.Next, n.NextHolding)
	}
	if err := b.checkOutcomes(n); err != nil {
		return err
	}
	if n.Mode == Sequential {
		return checkSequential(n)
	}
	return checkAbsolute(n)
}

func (b *builder) checkOutcomes(n Node) error {
	for _, o := range n.Outcomes {
		if _, ok := b.nodeIndex[o.To]; !ok {
			return modelError("outcome", n.ID+"→"+o.To, "target node not found")
		}
		if err := checkHolding(n.ID+"→"+o.To, o.Holding); err != nil {
			return err
		}
		if o.P != nil {
			if err := checkSlot("probability", n.ID+"→"+o.To, o.P, 0, 1, false); err != nil {
				return err
			}
		}
	}
	return nil
}

// checkSequential: every outcome but the last has a probability, the last
// has none.
func checkSequential(n Node) error {
	for i, o := range n.Outcomes {
		last := i == len(n.Outcomes)-1
		if last && o.P != nil {
			return modelError("node", n.ID, "sequential mode: the last outcome takes the remainder and has no probability")
		}
		if !last && o.P == nil {
			return modelError("node", n.ID, "sequential mode: every outcome but the last needs a probability")
		}
	}
	return nil
}

// checkAbsolute: either every probability is fixed and they sum to 1 with
// at most one remainder, or there are exactly two outcomes, one uncertain
// and one remainder.
func checkAbsolute(n Node) error {
	if len(n.Outcomes) == 0 {
		return nil
	}
	uncertain, remainder := 0, 0
	sum := 0.0
	for _, o := range n.Outcomes {
		switch {
		case o.P == nil:
			remainder++
		case isFixed(o.P):
			sum += o.P.Mean()
		default:
			uncertain++
		}
	}
	if uncertain > 0 {
		if len(n.Outcomes) == 2 && uncertain == 1 && remainder == 1 {
			return nil
		}
		return modelError("node", n.ID, "an uncertain probability in absolute mode needs exactly two outcomes, one uncertain and one remainder; use sequential mode otherwise")
	}
	switch {
	case remainder > 1:
		return modelError("node", n.ID, "more than one outcome takes the remainder")
	case remainder == 0 && math.Abs(sum-1) > probSumTol:
		return modelError("node", n.ID, "fixed probabilities sum to "+ftoa(sum)+", not 1")
	case remainder == 1 && sum > 1+probSumTol:
		return modelError("node", n.ID, "fixed probabilities exceed 1")
	}
	return nil
}

// isFixed reports a point mass: a fixed probability is any distribution
// whose support is one value.
func isFixed(d dist.Distribution) bool {
	lo, hi := d.Support()
	return lo == hi
}

// checkSlot validates a distribution against the support its slot
// requires and a finite mean.
func checkSlot(kind, subject string, d dist.Distribution, lo, hi float64, allowNil bool) error {
	if d == nil {
		if allowNil {
			return nil
		}
		return modelError(kind, subject, "no distribution")
	}
	slo, shi := d.Support()
	if slo < lo || shi > hi {
		return modelError(kind, subject, "support ["+ftoa(slo)+", "+ftoa(shi)+"] does not fit ["+ftoa(lo)+", "+ftoa(hi)+"]")
	}
	if math.IsInf(d.Mean(), 0) || math.IsNaN(d.Mean()) {
		return modelError(kind, subject, "infinite mean")
	}
	return nil
}

// checkHolding validates a holding time: non-negative support and a
// finite mean. Its quantile function, needed for the path-duration bound,
// is guaranteed by the type.
func checkHolding(subject string, d dist.Continuous) error {
	if d == nil {
		return nil
	}
	return checkSlot("holding", subject, d, 0, math.Inf(1), true)
}

// checkAcyclic runs Kahn's algorithm; a cycle is named.
func (b *builder) checkAcyclic() error {
	n := len(b.g.Nodes)
	indeg := make([]int, n)
	for _, nd := range b.g.Nodes {
		for _, to := range b.targets(nd) {
			indeg[to]++
		}
	}
	queue := make([]int32, 0, n)
	for v := range indeg {
		if indeg[v] == 0 {
			queue = append(queue, index(v))
		}
	}
	done := 0
	// bound: each node is dequeued once
	for i := 0; i < len(queue); i++ {
		done++
		for _, to := range b.targets(b.g.Nodes[queue[i]]) {
			indeg[to]--
			if indeg[to] == 0 {
				queue = append(queue, to)
			}
		}
	}
	if done == n {
		return nil
	}
	return modelError("graph", b.g.ID, "cycle: "+b.nameCycle(indeg))
}

// nameCycle walks the remaining nodes until one repeats.
func (b *builder) nameCycle(indeg []int) string {
	start := int32(-1)
	for v, d := range indeg {
		if d > 0 {
			start = index(v)
			break
		}
	}
	seen := map[int32]bool{}
	path := []string{}
	v := start
	// bound: a node repeats within len(nodes) steps
	for step := 0; step <= len(b.g.Nodes); step++ {
		if seen[v] {
			break
		}
		seen[v] = true
		path = append(path, b.g.Nodes[v].ID)
		next := int32(-1)
		for _, to := range b.targets(b.g.Nodes[v]) {
			if indeg[to] > 0 {
				next = to
				break
			}
		}
		if next < 0 {
			break
		}
		v = next
	}
	s := ""
	for i, id := range path {
		if i > 0 {
			s += " -> "
		}
		s += id
	}
	return s + " -> " + b.g.Nodes[v].ID
}

func (b *builder) targets(n Node) []int32 {
	out := make([]int32, 0, len(n.Outcomes)+1)
	for _, o := range n.Outcomes {
		out = append(out, b.nodeIndex[o.To])
	}
	if n.Next != "" {
		out = append(out, b.nodeIndex[n.Next])
	}
	return out
}

func (b *builder) checkWatch() error {
	for _, id := range append(append([]string{}, b.cfg.Watch.Nodes...), b.cfg.Watch.Leaves...) {
		if _, ok := b.nodeIndex[id]; !ok {
			return modelError("watch", id, "not a node")
		}
	}
	return nil
}
