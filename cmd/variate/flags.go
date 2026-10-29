package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// Output and record formats.
const (
	formatTable = "table"
	formatJSON  = "json"
	formatCSV   = "csv"
	formatJSONL = "jsonl"
	formatBin   = "bin"
)

// options is one invocation's flags.
type options struct {
	file       string
	settings   map[string]string // settings given on the command line, by file key
	periods    int64
	hasPeriods bool
	analytic   bool
	format     string
	out        string
	exceedance string
	rounds     string
	roundsOut  string
	watch      string
	leaf       string
	minValue   float64
}

// parseFlags reads the flags of plan, run and stream.
func parseFlags(name string, args []string, stderr io.Writer) (*options, error) {
	fs := flag.NewFlagSet("variate "+name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	o := &options{settings: map[string]string{}}
	fs.StringVar(&o.file, "f", "", "model file")
	values := map[string]*string{}
	bools := map[string]*bool{}
	for _, key := range config.Settings() {
		name := strings.ReplaceAll(key, "_", "-")
		if isBoolSetting(key) {
			bools[key] = fs.Bool(name, false, "run setting "+key)
		} else {
			values[key] = fs.String(name, "", "run setting "+key)
		}
	}
	fs.Int64Var(&o.periods, "n", 0, "fixed run length in periods")
	fs.BoolVar(&o.analytic, "analytic", false, "print the exact expectations and the transform")
	fs.StringVar(&o.format, "format", formatTable, "table, json or csv")
	fs.StringVar(&o.out, "out", "", "write the summary here")
	fs.StringVar(&o.exceedance, "exceedance", "", "write the exceedance curve here as CSV")
	fs.StringVar(&o.rounds, "rounds", "", "round record format: jsonl, csv or bin")
	fs.StringVar(&o.roundsOut, "rounds-out", "", "write round records here")
	fs.StringVar(&o.watch, "watch", "", "deliver rounds reaching any of these nodes (comma-separated)")
	fs.StringVar(&o.leaf, "leaf", "", "deliver rounds ending at any of these leaves")
	fs.Float64Var(&o.minValue, "min-value", 0, "deliver rounds booking at least this value")
	if err := fs.Parse(args); err != nil {
		return nil, fmt.Errorf("flags: %w", err)
	}
	if fs.NArg() > 0 {
		return nil, errors.New("unexpected argument " + fs.Arg(0))
	}
	if o.file == "" {
		return nil, errors.New("-f model file is required")
	}
	fs.Visit(func(f *flag.Flag) {
		key := strings.ReplaceAll(f.Name, "-", "_")
		if v, ok := values[key]; ok {
			o.settings[key] = *v
		}
		if b, ok := bools[key]; ok {
			o.settings[key] = strconv.FormatBool(*b)
		}
		if f.Name == "n" {
			o.hasPeriods = true
		}
	})
	switch o.format {
	case formatTable, formatJSON, formatCSV:
	default:
		return nil, errors.New("--format must be table, json or csv")
	}
	switch o.rounds {
	case "", formatJSONL, formatCSV, formatBin:
	default:
		return nil, errors.New("--rounds must be jsonl, csv or bin")
	}
	return o, nil
}

// apply loads the model and lays the flags over its settings.
func (o *options) apply() (*config.Model, error) {
	m, err := config.Load(o.file)
	if err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(o.settings))
	// unordered: the keys are sorted before use
	for key := range o.settings {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if err := config.SetSetting(&m.Config, key, o.settings[key]); err != nil {
			return nil, err
		}
	}
	if o.hasPeriods {
		m.Config.Periods = o.periods
	}
	if o.analytic {
		m.Config.Transform = true
	}
	m.Config.Watch = variate.Watch{Nodes: splitList(o.watch), Leaves: splitList(o.leaf), MinValue: o.minValue}
	return m, nil
}

// isBoolSetting reports whether a setting is a switch.
func isBoolSetting(key string) bool {
	switch key {
	case "allow_unconverged", "transform", "series", "no_breakdown", "node_hist":
		return true
	}
	return false
}

func splitList(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}
