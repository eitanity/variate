// Command norecursion checks that the main module has no direct or
// indirect recursion. It builds a class-hierarchy call graph of
// the given packages and reports every strongly connected component with
// more than one function, or a function that calls itself.
//
//	norecursion [-wrappers file] ./...
//
// A wrapper type that holds the interface it implements (a truncation over
// a distribution, say) looks recursive to class-hierarchy analysis, because
// an interface call on the inner value resolves to the wrapper itself. The
// -wrappers file lists such receiver types, one per line; interface
// dispatch into their methods is not followed. Static calls always are.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/callgraph/cha"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

const modulePath = "code.eitanity.com/variate"

func main() {
	wrappers := flag.String("wrappers", "", "file listing wrapper receiver types, one per line")
	flag.Parse()
	if err := run(flag.Args(), *wrappers); err != nil {
		fmt.Fprintln(os.Stderr, "norecursion:", err)
		os.Exit(2)
	}
}

func run(patterns []string, wrappersFile string) error {
	if len(patterns) == 0 {
		patterns = []string{"./..."}
	}
	wrappers, err := readWrappers(wrappersFile)
	if err != nil {
		return err
	}
	cfg := &packages.Config{Mode: packages.LoadAllSyntax}
	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		return err
	}
	if packages.PrintErrors(pkgs) > 0 {
		return fmt.Errorf("packages had errors")
	}
	prog, _ := ssautil.AllPackages(pkgs, ssa.InstantiateGenerics)
	prog.Build()
	cg := cha.CallGraph(prog)
	cycles := findCycles(cg, wrappers)
	if len(cycles) == 0 {
		return nil
	}
	for _, c := range cycles {
		fmt.Printf("recursion: %s\n", strings.Join(c, " -> "))
	}
	os.Exit(1)
	return nil
}

// own reports whether fn is a non-synthetic function of the main module.
func own(fn *ssa.Function) bool {
	if fn == nil || fn.Pkg == nil || fn.Synthetic != "" {
		return false
	}
	p := fn.Pkg.Pkg.Path()
	if strings.HasPrefix(p, modulePath+"/internal/lint") {
		return false
	}
	return p == modulePath || strings.HasPrefix(p, modulePath+"/")
}

// readWrappers loads the receiver types whose methods are not followed
// through interface dispatch.
func readWrappers(path string) (map[string]bool, error) {
	out := map[string]bool{}
	if path == "" {
		return out, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line != "" && !strings.HasPrefix(line, "#") {
			out[line] = true
		}
	}
	return out, sc.Err()
}

// receiverType names the receiver of a method as "pkg/path.Type", without
// a pointer marker; "" for a function.
func receiverType(fn *ssa.Function) string {
	if fn.Signature.Recv() == nil {
		return ""
	}
	t := fn.Signature.Recv().Type().String()
	return strings.TrimPrefix(t, "*")
}

// findCycles returns the strongly connected components of size > 1 and the
// self-loops among own functions, as sorted lists of function names.
// Interface-dispatched edges into a listed wrapper type are not followed.
func findCycles(cg *callgraph.Graph, wrappers map[string]bool) [][]string {
	adj := map[*ssa.Function][]*ssa.Function{}
	for fn, node := range cg.Nodes {
		if !own(fn) {
			continue
		}
		for _, e := range node.Out {
			callee := e.Callee.Func
			if !own(callee) {
				continue
			}
			if e.Site != nil && e.Site.Common().IsInvoke() && wrappers[receiverType(callee)] {
				continue
			}
			adj[fn] = append(adj[fn], callee)
		}
	}
	var out [][]string
	for _, comp := range tarjan(adj) {
		if len(comp) > 1 || selfLoop(adj, comp[0]) {
			names := make([]string, 0, len(comp))
			for _, fn := range comp {
				names = append(names, fn.String())
			}
			sort.Strings(names)
			out = append(out, names)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i][0] < out[j][0] })
	return out
}

func selfLoop(adj map[*ssa.Function][]*ssa.Function, fn *ssa.Function) bool {
	for _, c := range adj[fn] {
		if c == fn {
			return true
		}
	}
	return false
}

// tarjan is an iterative Tarjan SCC over adj.
func tarjan(adj map[*ssa.Function][]*ssa.Function) [][]*ssa.Function {
	index := map[*ssa.Function]int{}
	low := map[*ssa.Function]int{}
	onStack := map[*ssa.Function]bool{}
	var stack []*ssa.Function
	var comps [][]*ssa.Function
	next := 0
	type frame struct {
		fn *ssa.Function
		i  int
	}
	for root := range adj {
		if _, seen := index[root]; seen {
			continue
		}
		work := []frame{{root, 0}}
		index[root], low[root] = next, next
		next++
		stack = append(stack, root)
		onStack[root] = true
		for len(work) > 0 {
			f := &work[len(work)-1]
			if f.i < len(adj[f.fn]) {
				w := adj[f.fn][f.i]
				f.i++
				if _, seen := index[w]; !seen {
					index[w], low[w] = next, next
					next++
					stack = append(stack, w)
					onStack[w] = true
					work = append(work, frame{w, 0})
				} else if onStack[w] && index[w] < low[f.fn] {
					low[f.fn] = index[w]
				}
				continue
			}
			if low[f.fn] == index[f.fn] {
				var comp []*ssa.Function
				for {
					w := stack[len(stack)-1]
					stack = stack[:len(stack)-1]
					onStack[w] = false
					comp = append(comp, w)
					if w == f.fn {
						break
					}
				}
				comps = append(comps, comp)
			}
			done := f.fn
			work = work[:len(work)-1]
			if len(work) > 0 {
				parent := work[len(work)-1].fn
				if low[done] < low[parent] {
					low[parent] = low[done]
				}
			}
		}
	}
	return comps
}
