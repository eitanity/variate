package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"code.eitanity.com/variate"
	"go.yaml.in/yaml/v3"
)

const (
	// maxFileSize is the cap on a model file, checked before parsing.
	maxFileSize = 16 << 20
	// maxDataSize is the cap on a data file named by the model.
	maxDataSize = 64 << 20
	// maxDepth caps template nesting; maxInstances caps the instances a
	// file expands to.
	maxDepth     = 8
	maxInstances = 1024
)

// Model is a loaded file: the merged graph, the run settings it declares,
// and the period length in hours (0 when the file declares none).
type Model struct {
	Graph       variate.Graph
	Config      variate.Config
	PeriodHours float64
	Blocks      []Block // the sources whose rates came from reliability blocks
}

// Load reads and parses a model file. Data files it names resolve
// relative to its directory.
func Load(path string) (*Model, error) {
	f, err := os.Open(path) //nolint:gosec // the model file is the one the caller names
	if err != nil {
		return nil, fmt.Errorf("open model: %w", err)
	}
	defer closeQuietly(f)
	data, err := io.ReadAll(io.LimitReader(f, maxFileSize+1))
	if err != nil {
		return nil, fmt.Errorf("read model: %w", err)
	}
	dir, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return nil, fmt.Errorf("model directory: %w", err)
	}
	return Parse(data, dir)
}

// Parse parses model text. dir is the directory data files resolve in;
// empty forbids them.
func Parse(data []byte, dir string) (*Model, error) {
	if len(data) > maxFileSize {
		return nil, errTooLarge
	}
	var doc yaml.Node
	dec := yaml.NewDecoder(bytes.NewReader(data))
	if err := dec.Decode(&doc); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, errors.New("model file is empty")
		}
		return nil, fmt.Errorf("parse model: %w", err)
	}
	if err := dec.Decode(&yaml.Node{}); !errors.Is(err, io.EOF) {
		return nil, errors.New("model file holds more than one document")
	}
	if doc.Kind != yaml.DocumentNode || len(doc.Content) != 1 {
		return nil, errors.New("model file is empty")
	}
	if err := filter(doc.Content[0]); err != nil {
		return nil, err
	}
	l := &loader{dir: dir}
	return l.load(node{n: doc.Content[0], path: "model"})
}

// maxNesting is the deepest nesting a model file may have.
const maxNesting = 32

// filter refuses the YAML features a model has no use for before the
// walker sees the tree: anchors, aliases, tags other than the core
// ones, and nesting beyond maxNesting. Each is named with its line.
func filter(root *yaml.Node) error {
	type item struct {
		n     *yaml.Node
		depth int
	}
	stack := []item{{root, 1}}
	// bound: nodes in the document
	for len(stack) > 0 {
		it := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		n := it.n
		switch {
		case n.Kind == yaml.AliasNode:
			return fmt.Errorf("line %d: aliases are not allowed in a model file", n.Line)
		case n.Anchor != "":
			return fmt.Errorf("line %d: anchors are not allowed in a model file", n.Line)
		case !coreTag(n.Tag):
			return fmt.Errorf("line %d: tag %q is not allowed in a model file", n.Line, n.Tag)
		case it.depth > maxNesting:
			return fmt.Errorf("line %d: nesting deeper than %d", n.Line, maxNesting)
		}
		for _, c := range n.Content {
			stack = append(stack, item{c, it.depth + 1})
		}
	}
	return nil
}

// coreTag accepts the tags the YAML core schema resolves on its own.
func coreTag(tag string) bool {
	switch tag {
	case "", "!!str", "!!int", "!!float", "!!bool", "!!null", "!!map", "!!seq":
		return true
	}
	return false
}

// loader holds one file's state.
type loader struct {
	dir         string
	periodHours float64
	templates   map[string]template
	instances   int
	model       Model
}

// topKeys are the keys allowed at the top level besides the settings.
func topKeys() []string {
	return append([]string{"period", "graphs", "templates", "factors", "schedules", "layers"}, settingKeys()...)
}

func (l *loader) load(root node) (*Model, error) {
	top, err := root.mapping(topKeys()...)
	if err != nil {
		return nil, err
	}
	if p, ok := top["period"]; ok {
		if l.periodHours, err = parsePeriod(p); err != nil {
			return nil, err
		}
	}
	if err := l.settings(top); err != nil {
		return nil, err
	}
	if t, ok := top["templates"]; ok {
		if err := l.parseTemplates(t); err != nil {
			return nil, err
		}
	}
	if f, ok := top["factors"]; ok {
		if err := l.parseFactors(f); err != nil {
			return nil, err
		}
	}
	if sc, ok := top["schedules"]; ok {
		if err := l.parseSchedules(sc); err != nil {
			return nil, err
		}
	}
	if y, ok := top["layers"]; ok {
		if err := l.parseLayers(y); err != nil {
			return nil, err
		}
	}
	graphs, ok := top["graphs"]
	if !ok {
		return nil, root.errorf("no graphs")
	}
	if err := l.parseGraphs(graphs); err != nil {
		return nil, err
	}
	l.model.PeriodHours = l.periodHours
	return &l.model, nil
}

// parseFactors reads the shared factors: a list of maps with an id and
// a distribution, each a multiplier on the rates of the sources naming
// it, drawn once per period.
func (l *loader) parseFactors(x node) error {
	items, err := x.items()
	if err != nil {
		return err
	}
	for _, it := range items {
		m, err := it.mapping(append([]string{"id", "description", "unit", "t", "truncate", "cap", "scope"}, families()...)...)
		if err != nil {
			return err
		}
		f := variate.Factor{}
		if f.ID, err = required(it, m, "id"); err != nil {
			return err
		}
		if f.Model, _, err = l.parseDist(it, slotFactor, "id", "description"); err != nil {
			return err
		}
		l.model.Graph.Factors = append(l.model.Graph.Factors, f)
	}
	return nil
}

// parseSchedules reads the schedules: a list of maps with an id and the
// cycle of multipliers, period p using the value at p modulo the
// cycle's length.
func (l *loader) parseSchedules(x node) error {
	items, err := x.items()
	if err != nil {
		return err
	}
	for _, it := range items {
		m, err := it.mapping("id", "description", "values")
		if err != nil {
			return err
		}
		sc := variate.Schedule{}
		if sc.ID, err = required(it, m, "id"); err != nil {
			return err
		}
		v, ok := m["values"]
		if !ok {
			return it.errorf("schedule %s has no values", sc.ID)
		}
		if sc.Values, err = v.floats(0); err != nil {
			return err
		}
		l.model.Graph.Schedules = append(l.model.Graph.Schedules, sc)
	}
	return nil
}

// parseLayers reads the layers of the period total: an id, the
// attachment, the limit (omitted for none) and the share (omitted for
// the whole band), each a number in the model's money.
func (l *loader) parseLayers(x node) error {
	items, err := x.items()
	if err != nil {
		return err
	}
	for _, it := range items {
		m, err := it.mapping("id", "attachment", "limit", "share")
		if err != nil {
			return err
		}
		y := variate.Layer{}
		if y.ID, err = required(it, m, "id"); err != nil {
			return err
		}
		// A fixed order, so that with two bad values the first error
		// reported is always the same one.
		for _, f := range [...]struct {
			key string
			dst *float64
		}{{"attachment", &y.Attachment}, {"limit", &y.Limit}, {"share", &y.Share}} {
			if v, ok := m[f.key]; ok {
				if *f.dst, err = v.float(); err != nil {
					return err
				}
			}
		}
		l.model.Graph.Layers = append(l.model.Graph.Layers, y)
	}
	return nil
}

// parsePeriod reads period: { hours | days | weeks: n } as hours.
func parsePeriod(x node) (float64, error) {
	m, err := x.mapping("hours", "days", "weeks")
	if err != nil {
		return 0, err
	}
	if len(m) != 1 {
		return 0, x.errorf("period needs exactly one of hours, days or weeks")
	}
	// unordered: the map has one entry
	for key, v := range m {
		n, err := v.float()
		if err != nil {
			return 0, err
		}
		if !(n > 0) || n > 1e9 {
			return 0, v.errorf("period length must be positive")
		}
		return n * unitHours(key), nil
	}
	return 0, nil
}

// unitHours is the length of an exact unit in hours.
func unitHours(unit string) float64 {
	switch unit {
	case "hours":
		return 1
	case "days":
		return 24
	case "weeks":
		return 168
	}
	return 0
}

// settingKeys lists the run settings a file may declare.
func settingKeys() []string {
	return []string{unitPeriods, "allow_unconverged", keyTransform, "transform_grid", "transform_tol", "transform_work", "transform_bytes", "min_hits", "contrib_rse", "target_rse",
		"tail", "tail_hits", "max_periods", "max_steps", "batch", "hist_res", "seed", "workers",
		"block_periods", "pool_size", "series", "no_breakdown", "node_hist", "spill", "max_memory",
		"pace", "round_scratch", "record_buffer"}
}

// settings fills Config from the top-level keys.
func (l *loader) settings(top map[string]node) error {
	c := &l.model.Config
	for _, key := range settingKeys() {
		v, ok := top[key]
		if !ok {
			continue
		}
		if err := setSetting(c, key, v); err != nil {
			return err
		}
	}
	return nil
}

// SetSetting assigns one run setting from its text form, as the file key
// (snake_case) or the flag name (kebab-case). Durations take Go syntax.
func SetSetting(c *variate.Config, key, value string) error {
	key = strings.ReplaceAll(key, "-", "_")
	if !contains(settingKeys(), key) {
		return fmt.Errorf("unknown setting %q", key)
	}
	return setSetting(c, key, node{n: &yaml.Node{Kind: yaml.ScalarNode, Value: value}, path: key})
}

// Settings lists the setting keys in file form.
func Settings() []string { return settingKeys() }

func setSetting(c *variate.Config, key string, v node) error {
	var err error
	switch key {
	case unitPeriods:
		c.Periods, err = v.integer(0, 1<<62)
	case "allow_unconverged":
		c.AllowUnconverged, err = v.boolean()
	case keyTransform, "transform_grid", "transform_tol", "transform_work", "transform_bytes":
		err = setTransformSetting(c, strings.TrimPrefix(key, keyTransform+"_"), v)
	case "min_hits":
		c.MinHits, err = uinteger(v)
	case "contrib_rse":
		c.ContribRSE, err = v.float()
	case "target_rse":
		c.TargetRSE, err = v.float()
	case "tail":
		c.TailP, err = v.float()
	case "tail_hits":
		c.TailHits, err = uinteger(v)
	case "max_periods":
		c.MaxPeriods, err = v.integer(0, 1<<62)
	case "max_steps":
		c.MaxSteps, err = uinteger(v)
	case "batch":
		c.Batch, err = smallInt(v)
	case "hist_res":
		c.HistRes, err = v.float()
	case "seed":
		c.Seed, err = uinteger(v)
	case "workers":
		c.Workers, err = smallInt(v)
	case "block_periods":
		c.BlockPeriods, err = smallInt(v)
	case "pool_size":
		c.PoolSize, err = smallInt(v)
	case "series":
		c.Series, err = v.boolean()
	case "no_breakdown":
		c.NoBreakdown, err = v.boolean()
	case "node_hist":
		c.NodeHist, err = v.boolean()
	case "spill":
		c.Spill, err = smallInt(v)
	case "max_memory":
		c.MaxMemory, err = v.integer(0, 1<<62)
	case "pace":
		c.Pace, err = duration(v)
	case "round_scratch":
		c.RoundScratch, err = smallInt(v)
	case "record_buffer":
		c.RecordBuffer, err = smallInt(v)
	}
	return err
}

const keyTransform = "transform"

// setTransformSetting applies one of the transform's settings, by the
// part of its key after "transform_" ("transform" itself for the switch).
func setTransformSetting(c *variate.Config, key string, v node) error {
	var err error
	switch key {
	case keyTransform:
		c.Transform, err = v.boolean()
	case "grid":
		c.TransformGrid, err = smallInt(v)
	case "tol":
		c.TransformTol, err = v.float()
	case "work":
		c.TransformWork, err = v.integer(0, 1<<62)
	default:
		c.TransformBytes, err = v.integer(0, 1<<62)
	}
	return err
}

// closeQuietly closes a file that was only read; a close error then
// carries no information the read did not.
func closeQuietly(f *os.File) { _ = f.Close() }

func uinteger(v node) (uint64, error) {
	n, err := v.integer(0, 1<<62)
	return uint64(n), err //nolint:gosec // bounded by the range check
}

func smallInt(v node) (int, error) {
	n, err := v.integer(0, 1<<31-1)
	return int(n), err
}

func duration(v node) (time.Duration, error) {
	s, err := v.str()
	if err != nil {
		return 0, err
	}
	d, err := time.ParseDuration(s)
	if err != nil || d < 0 {
		return 0, v.errorf("%q is not a duration", s)
	}
	return d, nil
}
