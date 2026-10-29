// Command variate is the generic command-line interface: it loads a
// model file, sizes or runs it, and reports the results as a table,
// JSON or CSV, with the rounds streamed on request.
package main

import (
	"errors"
	"io"
	"os"
	"runtime/debug"
)

const usage = `usage:
  variate version
  variate plan   -f model.yaml [settings] [--format table|json]
  variate run    -f model.yaml [settings] [-n PERIODS [--series]] [--analytic]
                 [--format table|json|csv] [--out FILE] [--exceedance FILE]
                 [--rounds jsonl|csv|bin --rounds-out FILE] [watch flags]
  variate stream -f model.yaml [run flags] [--watch NODE,...] [--leaf LEAF,...]
                 [--min-value X] [--pace DURATION] [--round-scratch N]
                 [--rounds jsonl|csv|bin] [--rounds-out FILE]

Every run setting of the model file has a flag of the same name in
kebab-case (--min-hits, --target-rse, --tail, --max-periods, ...), and
flags override the file. A seed of 0 picks a random seed, which is
printed. Example models live under examples/`

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

// run dispatches one invocation and returns the process exit code.
func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		return fail(stderr, errors.New(usage))
	}
	var err error
	switch args[0] {
	case "version":
		_, err = io.WriteString(stdout, version()+"\n")
	case "plan":
		err = plan(args[1:], stdout, stderr)
	case "run":
		err = runModel(args[1:], stdout, stderr, false)
	case "stream":
		err = runModel(args[1:], stdout, stderr, true)
	case "help", "-h", "--help":
		_, err = io.WriteString(stdout, usage+"\n")
	default:
		err = errors.New("unknown command " + args[0] + "; " + usage)
	}
	if err != nil {
		return fail(stderr, err)
	}
	return 0
}

// version reports the module version recorded at build time, or "devel".
func version() string {
	info, ok := debug.ReadBuildInfo()
	if !ok || info.Main.Version == "" {
		return "devel"
	}
	return info.Main.Version
}

func fail(stderr io.Writer, err error) int {
	_, _ = io.WriteString(stderr, "variate: "+err.Error()+"\n")
	return 1
}
