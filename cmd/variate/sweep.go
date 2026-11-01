package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/report"
)

// sweep loads the model, nudges each input down and up, and writes the
// tornado table: one row per input, the largest swing first.
func sweep(args []string, stdout, stderr io.Writer) error {
	o, err := parseFlags("sweep", args, stderr)
	if err != nil {
		return err
	}
	m, err := o.apply()
	if err != nil {
		return err
	}
	_, warnings, err := m.Graph.Compile(m.Config)
	if err != nil {
		return fmt.Errorf("compile: %w", err)
	}
	printWarnings(stderr, warnings)
	cfg := variate.SweepConfig{Figure: o.figure, Delta: o.delta, Inputs: splitList(o.inputs), Top: o.top, Run: m.Config}
	n := len(variate.SelectInputs(m.Graph.Inputs(), cfg.Inputs))
	if cfg.Top > 0 {
		n = min(n, cfg.Top)
	}
	fmt.Fprintf(stderr, "sweep: %d inputs × 2 nudges\n", n)
	sw, err := m.Graph.Sweep(context.Background(), cfg)
	if err != nil {
		return fmt.Errorf("sweep: %w", err)
	}
	dst := stdout
	closer := func() error { return nil }
	if o.out != "" {
		f, err := os.Create(o.out) //nolint:gosec // the output file the user named
		if err != nil {
			return fmt.Errorf("output file: %w", err)
		}
		dst, closer = f, func() error { return closeFile(f) }
	}
	bw := bufio.NewWriter(dst)
	switch o.format {
	case formatJSON:
		report.WriteSweepJSON(bw, sw)
	case formatCSV:
		report.WriteSweepCSV(bw, sw)
	default:
		report.WriteSweepTable(bw, sw)
	}
	if err := flush(bw); err != nil {
		return err
	}
	return closer()
}
