package config

import (
	"sort"
	"strconv"
	"strings"

	"code.eitanity.com/variate"
	"go.yaml.in/yaml/v3"
)

const keyNextHolding = "next_holding"

// template is a reusable sub-graph kept as nodes until it is instanced.
type template struct {
	id        string
	entry     string
	exits     []string
	nodes     node
	instances node
	hasInst   bool
	line      int
}

// instance is one expansion job: a template under a prefix, with its
// exits wired to node names in the merged graph.
type instance struct {
	tmpl      string
	as        string            // full node prefix, without the trailing dot
	wire      map[string]string // exit → merged node name
	overrides map[string]node   // path within the template → distribution
	depth     int
	at        node
}

// naming resolves node references while a graph or an instance is parsed.
type naming struct {
	prefix string            // prepended to every local name
	exits  map[string]string // template exits → merged names; nil in a graph
}

func (nm naming) resolve(x node, ref string) (string, error) {
	if nm.exits != nil {
		if target, ok := nm.exits[ref]; ok {
			return target, nil
		}
	}
	return nm.prefix + ref, nil
}

func (l *loader) parseTemplates(x node) error {
	items, err := x.items()
	if err != nil {
		return err
	}
	l.templates = map[string]template{}
	for _, it := range items {
		m, err := it.mapping("id", "entry", "exits", "nodes", "instances")
		if err != nil {
			return err
		}
		t := template{line: it.line()}
		if t.id, err = required(it, m, "id"); err != nil {
			return err
		}
		if t.entry, err = required(it, m, "entry"); err != nil {
			return err
		}
		if ex, ok := m["exits"]; ok {
			if t.exits, err = strList(ex); err != nil {
				return err
			}
		}
		nodes, ok := m["nodes"]
		if !ok {
			return it.errorf("template %q has no nodes", t.id)
		}
		t.nodes = nodes
		t.instances, t.hasInst = m["instances"]
		if _, dup := l.templates[t.id]; dup {
			return it.errorf("duplicate template %q", t.id)
		}
		l.templates[t.id] = t
	}
	return nil
}

func required(x node, m map[string]node, key string) (string, error) {
	v, ok := m[key]
	if !ok {
		return "", x.errorf("missing %s", key)
	}
	return v.str()
}

func strList(x node) ([]string, error) {
	items, err := x.items()
	if err != nil {
		return nil, err
	}
	out := make([]string, len(items))
	for i, it := range items {
		if out[i], err = it.str(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// parseGraphs reads every graph, prefixing names with the graph's ID
// when there is more than one, and expands their instances.
func (l *loader) parseGraphs(x node) error {
	items, err := x.items()
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return x.errorf("no graphs")
	}
	var jobs []instance
	for _, it := range items {
		more, err := l.parseGraph(it, len(items) > 1)
		if err != nil {
			return err
		}
		jobs = append(jobs, more...)
	}
	return l.expand(jobs)
}

// parseGraph reads one graph's sources, nodes and instances.
func (l *loader) parseGraph(it node, prefixed bool) ([]instance, error) {
	m, err := it.mapping("id", "description", "sources", "nodes", "instances")
	if err != nil {
		return nil, err
	}
	id, err := required(it, m, "id")
	if err != nil {
		return nil, err
	}
	nm := naming{}
	if prefixed {
		nm.prefix = id + "."
	} else {
		l.model.Graph.ID = id
	}
	if d, ok := m["description"]; ok {
		if l.model.Graph.Description, err = d.scalar(); err != nil {
			return nil, err
		}
	}
	if src, ok := m["sources"]; ok {
		if err := l.parseSources(src, nm); err != nil {
			return nil, err
		}
	}
	if nodes, ok := m["nodes"]; ok {
		if err := l.parseNodes(nodes, nm); err != nil {
			return nil, err
		}
	}
	if inst, ok := m["instances"]; ok {
		return l.parseInstances(inst, nm, 0)
	}
	return nil, nil
}

func (l *loader) parseSources(x node, nm naming) error {
	items, err := x.items()
	if err != nil {
		return err
	}
	for _, it := range items {
		m, err := it.mapping("id", "description", "frequency", "block", "to", "every", "repair", "factor", "schedule")
		if err != nil {
			return err
		}
		s := variate.Source{}
		if s.ID, err = required(it, m, "id"); err != nil {
			return err
		}
		s.ID = nm.prefix + s.ID
		if d, ok := m["description"]; ok {
			if s.Description, err = d.scalar(); err != nil {
				return err
			}
		}
		to, err := required(it, m, "to")
		if err != nil {
			return err
		}
		if s.To, err = nm.resolve(it, to); err != nil {
			return err
		}
		if err := l.sourceRate(it, m, &s); err != nil {
			return err
		}
		if err := l.sourceLaw(m, &s); err != nil {
			return err
		}
		l.model.Graph.Sources = append(l.model.Graph.Sources, s)
	}
	return nil
}

// sourceLaw reads a source's count law beyond Poisson: every n-th
// arrival, or a repair holding during which it is silent, and the
// shared factor and the schedule on its rate.
func (l *loader) sourceLaw(m map[string]node, s *variate.Source) error {
	if f, ok := m["factor"]; ok {
		var err error
		if s.Factor, err = f.str(); err != nil {
			return err
		}
	}
	if sc, ok := m["schedule"]; ok {
		var err error
		if s.Schedule, err = sc.str(); err != nil {
			return err
		}
	}
	if e, ok := m["every"]; ok {
		n, err := e.integer(1, 1<<20)
		if err != nil {
			return err
		}
		s.Every = int(n)
	}
	if r, ok := m["repair"]; ok {
		var err error
		if s.Repair, _, err = l.parseDist(r, slotHolding); err != nil {
			return err
		}
	}
	return nil
}

// sourceRate reads a source's frequency, or the reliability block that
// stands in for it.
func (l *loader) sourceRate(it node, m map[string]node, s *variate.Source) error {
	f, hasF := m["frequency"]
	b, hasB := m["block"]
	switch {
	case hasF && hasB:
		return it.errorf("a source has frequency or block, not both")
	case hasF:
		var err error
		s.Frequency, _, err = l.parseDist(f, slotFrequency)
		return err
	case hasB:
		if _, ok := m["every"]; ok {
			return it.errorf("a block takes no every")
		}
		if _, ok := m["repair"]; ok {
			return it.errorf("a block takes no repair")
		}
		return l.blockSource(s.ID, b, s)
	default:
		return it.errorf("missing frequency")
	}
}

func (l *loader) parseNodes(x node, nm naming) error {
	items, err := x.items()
	if err != nil {
		return err
	}
	for _, it := range items {
		n, err := l.parseNode(it, nm)
		if err != nil {
			return err
		}
		l.model.Graph.Nodes = append(l.model.Graph.Nodes, n)
	}
	return nil
}

func (l *loader) parseNode(it node, nm naming) (variate.Node, error) {
	m, err := it.mapping("id", "description", "functions", "mode", "outcomes", "next", keyNextHolding)
	if err != nil {
		return variate.Node{}, err
	}
	n := variate.Node{}
	if n.ID, err = required(it, m, "id"); err != nil {
		return n, err
	}
	n.ID = nm.prefix + n.ID
	if d, ok := m["description"]; ok {
		if n.Description, err = d.scalar(); err != nil {
			return n, err
		}
	}
	if mode, ok := m["mode"]; ok {
		s, err := mode.oneOf("absolute", "sequential")
		if err != nil {
			return n, err
		}
		if s == "sequential" {
			n.Mode = variate.Sequential
		}
	}
	if fns, ok := m["functions"]; ok {
		if n.Functions, err = l.parseFunctions(fns); err != nil {
			return n, err
		}
	}
	if outs, ok := m["outcomes"]; ok {
		if n.Outcomes, err = l.parseOutcomes(outs, nm); err != nil {
			return n, err
		}
	}
	return l.parseNext(n, m, nm)
}

// parseNext reads next and next_holding.
func (l *loader) parseNext(n variate.Node, m map[string]node, nm naming) (variate.Node, error) {
	next, ok := m["next"]
	nh, hasHolding := m[keyNextHolding]
	if !ok {
		if hasHolding {
			return n, nh.errorf("next_holding without next")
		}
		return n, nil
	}
	ref, err := next.str()
	if err != nil {
		return n, err
	}
	if n.Next, err = nm.resolve(next, ref); err != nil {
		return n, err
	}
	if hasHolding {
		if s, serr := nh.str(); serr == nil && s == "outage" {
			n.NextOutage = true
		} else if n.NextHolding, n.NextScope, err = l.parseDist(nh, slotHolding); err != nil {
			return n, err
		}
	}
	return n, nil
}

func (l *loader) parseFunctions(x node) ([]variate.Function, error) {
	items, err := x.items()
	if err != nil {
		return nil, err
	}
	out := make([]variate.Function, 0, len(items))
	for _, it := range items {
		f, err := l.parseFunction(it)
		if err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	return out, nil
}

func (l *loader) parseFunction(it node) (variate.Function, error) {
	m, err := it.entries()
	if err != nil {
		return variate.Function{}, err
	}
	f := variate.Function{}
	s := slotValue
	for _, e := range m {
		switch e.key {
		case "name":
			f.Name, err = e.value.str()
		case "category":
			f.Category, err = e.value.str()
		case "basis":
			var b string
			if b, err = e.value.oneOf("on_entry", "per_time"); b == "per_time" {
				f.Basis, s = variate.PerTime, slotRate
			}
		}
		if err != nil {
			return f, err
		}
	}
	if f.Name == "" || f.Category == "" {
		return f, it.errorf("a function needs name and category")
	}
	f.Model, f.Scope, err = l.parseDist(it, s, "name", "category", "basis")
	return f, err
}

func (l *loader) parseOutcomes(x node, nm naming) ([]variate.Outcome, error) {
	items, err := x.items()
	if err != nil {
		return nil, err
	}
	out := make([]variate.Outcome, 0, len(items))
	for _, it := range items {
		m, err := it.mapping("to", "p", "holding")
		if err != nil {
			return nil, err
		}
		o := variate.Outcome{}
		to, err := required(it, m, "to")
		if err != nil {
			return nil, err
		}
		if o.To, err = nm.resolve(it, to); err != nil {
			return nil, err
		}
		if p, ok := m["p"]; ok {
			if o.P, _, err = l.parseDist(p, slotProbability); err != nil {
				return nil, err
			}
		}
		if h, ok := m["holding"]; ok {
			if s, serr := h.str(); serr == nil && s == "outage" {
				o.OutageHolding = true
			} else if o.Holding, o.HoldingScope, err = l.parseDist(h, slotHolding); err != nil {
				return nil, err
			}
		}
		out = append(out, o)
	}
	return out, nil
}

// parseInstances reads an instances list into expansion jobs. Wire
// targets and nested names are resolved through nm.
func (l *loader) parseInstances(x node, nm naming, depth int) ([]instance, error) {
	items, err := x.items()
	if err != nil {
		return nil, err
	}
	jobs := make([]instance, 0, len(items))
	for _, it := range items {
		job, err := l.parseInstance(it, nm, depth)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, job)
	}
	return jobs, nil
}

func (l *loader) parseInstance(it node, nm naming, depth int) (instance, error) {
	m, err := it.mapping("template", "as", "wire", "override")
	if err != nil {
		return instance{}, err
	}
	job := instance{depth: depth, at: it, wire: map[string]string{}, overrides: map[string]node{}}
	if job.tmpl, err = required(it, m, "template"); err != nil {
		return job, err
	}
	as, err := required(it, m, "as")
	if err != nil {
		return job, err
	}
	job.as = nm.prefix + as
	if w, ok := m["wire"]; ok {
		if err := parseWire(w, nm, job.wire); err != nil {
			return job, err
		}
	}
	if o, ok := m["override"]; ok {
		es, err := o.entries()
		if err != nil {
			return job, err
		}
		for _, e := range es {
			job.overrides[e.key] = e.value
		}
	}
	return job, nil
}

// parseWire reads exit: target pairs, resolving targets in nm.
func parseWire(w node, nm naming, wire map[string]string) error {
	es, err := w.entries()
	if err != nil {
		return err
	}
	for _, e := range es {
		ref, err := e.value.str()
		if err != nil {
			return err
		}
		if wire[e.key], err = nm.resolve(e.value, ref); err != nil {
			return err
		}
	}
	return nil
}

// expand works through the instance jobs, each adding its template's
// nodes under its prefix and queueing the template's own instances.
func (l *loader) expand(jobs []instance) error {
	// bound: maxInstances
	for len(jobs) > 0 {
		job := jobs[0]
		jobs = jobs[1:]
		l.instances++
		if l.instances > maxInstances {
			return job.at.errorf("more than %d instances; template %q", maxInstances, job.tmpl)
		}
		if job.depth >= maxDepth {
			return job.at.errorf("template %q nested deeper than %d", job.tmpl, maxDepth)
		}
		t, ok := l.templates[job.tmpl]
		if !ok {
			return job.at.errorf("unknown template %q", job.tmpl)
		}
		more, err := l.instantiate(t, job)
		if err != nil {
			return err
		}
		jobs = append(jobs, more...)
	}
	return nil
}

// instantiate parses one copy of a template's nodes.
func (l *loader) instantiate(t template, job instance) ([]instance, error) {
	for _, exit := range t.exits {
		if _, ok := job.wire[exit]; !ok {
			return nil, job.at.errorf("exit %q of template %q is not wired", exit, job.tmpl)
		}
	}
	for _, exit := range sortedPaths(job.wire) {
		if !contains(t.exits, exit) {
			return nil, job.at.errorf("template %q has no exit %q", job.tmpl, exit)
		}
	}
	nm := naming{prefix: job.as + ".", exits: job.wire}
	nodes := node{n: clone(t.nodes.n), path: "template " + t.id + " as " + job.as + ".nodes"}
	own, nested, err := splitOverrides(job)
	if err != nil {
		return nil, err
	}
	if err := applyOverrides(nodes, own); err != nil {
		return nil, err
	}
	if err := l.parseNodes(nodes, nm); err != nil {
		return nil, err
	}
	if !t.hasInst {
		if len(nested) > 0 {
			return nil, job.at.errorf("override path names an instance but template %q has none", t.id)
		}
		return nil, nil
	}
	more, err := l.parseInstances(t.instances, nm, job.depth+1)
	if err != nil {
		return nil, err
	}
	for i := range more {
		// unordered: a map copied into a map, each key to its own
		for path, v := range nested[strings.TrimPrefix(more[i].as, nm.prefix)] {
			more[i].overrides[path] = v
		}
	}
	return more, nil
}

// splitOverrides separates overrides of the template's own nodes from
// those addressed to a nested instance (first segment = its name).
func splitOverrides(job instance) (own map[string]node, nested map[string]map[string]node, err error) {
	own = map[string]node{}
	nested = map[string]map[string]node{}
	for _, path := range sortedPaths(job.overrides) {
		v := job.overrides[path]
		head, rest, found := strings.Cut(path, ".")
		if !found {
			return nil, nil, v.errorf("override path %q is too short", path)
		}
		if strings.Contains(head, "[") {
			return nil, nil, v.errorf("override path %q must start with a node name", path)
		}
		switch {
		case strings.HasPrefix(rest, "outcomes[") || strings.HasPrefix(rest, "functions[") || rest == keyNextHolding:
			own[path] = v
		default:
			if nested[head] == nil {
				nested[head] = map[string]node{}
			}
			nested[head][rest] = v
		}
	}
	return own, nested, nil
}

// applyOverrides replaces distributions in a cloned nodes list by path:
// <node>.outcomes[i].p, <node>.outcomes[i].holding, <node>.functions[i]
// (the family and modifiers, keeping name, category and basis) or
// <node>.next_holding.
func applyOverrides(nodes node, overrides map[string]node) error {
	for _, path := range sortedPaths(overrides) {
		v := overrides[path]
		nodeID, rest, _ := strings.Cut(path, ".")
		target := findNode(nodes, nodeID)
		if target == nil {
			return v.errorf("override path %q names no node", path)
		}
		if err := override(target, rest, v, path); err != nil {
			return err
		}
	}
	return nil
}

func findNode(nodes node, id string) *yaml.Node {
	for _, n := range deref(nodes.n).Content {
		n = deref(n)
		for i := 0; i+1 < len(n.Content); i += 2 {
			if deref(n.Content[i]).Value == "id" && deref(n.Content[i+1]).Value == id {
				return n
			}
		}
	}
	return nil
}

// mapValue returns the value node of a key in a mapping node, or nil.
func mapValue(m *yaml.Node, key string) *yaml.Node {
	m = deref(m)
	for i := 0; i+1 < len(m.Content); i += 2 {
		if deref(m.Content[i]).Value == key {
			return m.Content[i+1]
		}
	}
	return nil
}

func override(target *yaml.Node, rest string, v node, path string) error {
	key, idx, field, err := splitPath(rest)
	if err != nil {
		return v.errorf("override path %q: %v", path, err)
	}
	list := mapValue(target, key)
	if list == nil {
		return v.errorf("override path %q: node has no %s", path, key)
	}
	if key == keyNextHolding {
		*list = *clone(v.n)
		return nil
	}
	items := deref(list).Content
	if idx < 0 || idx >= len(items) {
		return v.errorf("override path %q: index out of range", path)
	}
	item := deref(items[idx])
	if key == "functions" {
		replaceFamily(item, v)
		return nil
	}
	slot := mapValue(item, field)
	if slot == nil {
		return v.errorf("override path %q: outcome has no %s", path, field)
	}
	*slot = *clone(v.n)
	return nil
}

// splitPath parses "outcomes[2].p", "functions[0]" or "next_holding".
func splitPath(rest string) (key string, idx int, field string, err error) {
	if rest == keyNextHolding {
		return rest, 0, "", nil
	}
	open := strings.Index(rest, "[")
	closeIdx := strings.Index(rest, "]")
	if open < 0 || closeIdx < open {
		return "", 0, "", errNotAPath
	}
	key = rest[:open]
	idx, err = strconv.Atoi(rest[open+1 : closeIdx])
	if err != nil {
		return "", 0, "", errNotAPath
	}
	switch key {
	case "functions":
		if closeIdx != len(rest)-1 {
			return "", 0, "", errNotAPath
		}
	case "outcomes":
		field = strings.TrimPrefix(rest[closeIdx+1:], ".")
		if field != "p" && field != "holding" {
			return "", 0, "", errNotAPath
		}
	default:
		return "", 0, "", errNotAPath
	}
	return key, idx, field, nil
}

// replaceFamily rewrites a function map's distribution keys from v,
// keeping name, category and basis.
func replaceFamily(fn *yaml.Node, v node) {
	kept := make([]*yaml.Node, 0, len(fn.Content))
	for i := 0; i+1 < len(fn.Content); i += 2 {
		k := deref(fn.Content[i]).Value
		if k == "name" || k == "category" || k == "basis" {
			kept = append(kept, fn.Content[i], fn.Content[i+1])
		}
	}
	src := deref(v.n)
	if src.Kind == yaml.ScalarNode {
		kept = append(kept, &yaml.Node{Kind: yaml.ScalarNode, Value: famConstant}, clone(src))
	} else {
		kept = append(kept, clone(src).Content...)
	}
	fn.Content = kept
}

// sortedPaths lists a map's keys in order, so that the first of several
// errors reported is always the same one.
func sortedPaths[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	// unordered: the keys are sorted before use
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
