package analyzers

import (
	"go/ast"
	"go/token"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// LoopBound checks the engine packages: every for statement that is not a range loop and whose condition is not a
// comparison against a length, a constant or a fixed bound carries a
// "// bound:" comment.
var LoopBound = &analysis.Analyzer{
	Name: "loopbound",
	Doc:  "every loop in the root package, dist and stats states its bound",
	Run:  runLoopBound,
}

func runLoopBound(pass *analysis.Pass) (any, error) {
	if !inEngine(pass) {
		return nil, nil
	}
	for _, f := range pass.Files {
		if isTest(pass, f.Pos()) {
			continue
		}
		cmap := ast.NewCommentMap(pass.Fset, f, f.Comments)
		ast.Inspect(f, func(n ast.Node) bool {
			fs, ok := n.(*ast.ForStmt)
			if !ok {
				return true
			}
			if boundedCond(pass, fs.Cond) || hasBoundComment(pass, cmap, fs) {
				return true
			}
			pass.Reportf(fs.Pos(), "loop without a stated bound: add a // bound: comment or a fixed condition")
			return true
		})
	}
	return nil, nil
}

// boundedCond accepts `i < len(x)`, `i < N` for a constant N, and the same
// with <=, !=, > and >= in either orientation.
func boundedCond(pass *analysis.Pass, cond ast.Expr) bool {
	be, ok := cond.(*ast.BinaryExpr)
	if !ok {
		return false
	}
	switch be.Op {
	case token.LSS, token.LEQ, token.GTR, token.GEQ, token.NEQ:
	default:
		return false
	}
	return isBoundExpr(pass, be.X) || isBoundExpr(pass, be.Y)
}

// isBoundExpr reports whether e is len(...), cap(...) or a constant.
func isBoundExpr(pass *analysis.Pass, e ast.Expr) bool {
	if call, ok := e.(*ast.CallExpr); ok {
		if id, ok := call.Fun.(*ast.Ident); ok && (id.Name == "len" || id.Name == "cap") {
			return true
		}
	}
	tv, ok := pass.TypesInfo.Types[e]
	return ok && tv.Value != nil
}

// hasBoundComment looks for "bound:" in a comment attached to the loop or
// on the line just above it.
func hasBoundComment(pass *analysis.Pass, cmap ast.CommentMap, fs *ast.ForStmt) bool {
	for _, cg := range cmap[fs] {
		if strings.Contains(cg.Text(), "bound:") {
			return true
		}
	}
	line := pass.Fset.Position(fs.Pos()).Line
	for _, cg := range cmap.Comments() {
		end := pass.Fset.Position(cg.End()).Line
		if (end == line-1 || end == line) && strings.Contains(cg.Text(), "bound:") {
			return true
		}
	}
	return false
}
