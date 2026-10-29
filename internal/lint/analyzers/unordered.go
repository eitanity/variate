package analyzers

import (
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// Unordered checks every package of the module outside tests: a range
// over a map carries an "// unordered:" comment saying why the
// iteration order cannot matter, since Go randomises it; otherwise the
// keys are to be sorted first. A result, an error message or an
// output that depends on the order would differ run to run.
var Unordered = &analysis.Analyzer{
	Name: "unordered",
	Doc:  "a range over a map states why its order cannot matter",
	Run:  runUnordered,
}

func runUnordered(pass *analysis.Pass) (any, error) {
	for _, f := range pass.Files {
		if isTest(pass, f.Pos()) {
			continue
		}
		cmap := ast.NewCommentMap(pass.Fset, f, f.Comments)
		ast.Inspect(f, func(n ast.Node) bool {
			rs, ok := n.(*ast.RangeStmt)
			if !ok {
				return true
			}
			t := pass.TypesInfo.TypeOf(rs.X)
			if t == nil {
				return true
			}
			if _, isMap := t.Underlying().(*types.Map); !isMap || hasComment(pass, cmap, rs, "unordered:") {
				return true
			}
			pass.Reportf(rs.Pos(), "range over a map without a stated reason that its order cannot matter: add an // unordered: comment or iterate sorted keys")
			return true
		})
	}
	return nil, nil
}

// hasComment looks for a marker in a comment attached to the statement
// or on the line just above it.
func hasComment(pass *analysis.Pass, cmap ast.CommentMap, n ast.Node, marker string) bool {
	for _, cg := range cmap[n] {
		if strings.Contains(cg.Text(), marker) {
			return true
		}
	}
	line := pass.Fset.Position(n.Pos()).Line
	for _, cg := range cmap.Comments() {
		end := pass.Fset.Position(cg.End()).Line
		if (end == line-1 || end == line) && strings.Contains(cg.Text(), marker) {
			return true
		}
	}
	return false
}
