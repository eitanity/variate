package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
	"code.eitanity.com/variate/report"
)

// compare loads the model and each option's model file and writes each
// option's figure against the model's, run with the model's settings at
// its one seed.
func compare(args []string, stdout, stderr io.Writer) error {
	o, err := parseFlags("compare", args, stderr)
	if err != nil {
		return err
	}
	if len(o.options) == 0 {
		return errors.New("--option model file is required")
	}
	m, err := o.apply()
	if err != nil {
		return err
	}
	m.Config.PickSeed()
	_, warnings, err := m.Graph.Compile(m.Config)
	if err != nil {
		return fmt.Errorf("compile: %w", err)
	}
	printWarnings(stderr, warnings)
	opts := make([]variate.Graph, len(o.options))
	for i, f := range o.options {
		om, err := config.Load(f)
		if err != nil {
			return fmt.Errorf("option %s: %w", f, err)
		}
		om.Phase = m.Phase
		if err := om.ApplyPhase(); err != nil {
			return fmt.Errorf("option %s: %w", f, err)
		}
		opts[i] = om.Graph
	}
	if len(opts) == 1 {
		fmt.Fprintln(stderr, "compare: 1 option")
	} else {
		fmt.Fprintf(stderr, "compare: %d options\n", len(opts))
	}
	c, err := m.Graph.Compare(context.Background(), opts, variate.CompareConfig{Figure: o.figure, Run: m.Config})
	if err != nil {
		return fmt.Errorf("%w", err) // the library's errors name the comparison
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
		report.WriteCompareJSON(bw, c, o.options)
	case formatCSV:
		report.WriteCompareCSV(bw, c, o.options)
	default:
		report.WriteCompareTable(bw, c, o.options)
	}
	if err := flush(bw); err != nil {
		return err
	}
	return closer()
}
