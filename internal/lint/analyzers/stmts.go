package analyzers

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
)

// Stmts enforces the statement bans: no goto, recover only in the root
// package's stream.go, and go statements only in its workspace.go.
var Stmts = &analysis.Analyzer{
	Name: "stmts",
	Doc:  "no goto; recover only around Observe; goroutines only in the Workspace constructor",
	Run:  runStmts,
}

const (
	goFile      = "workspace.go"
	recoverFile = "stream.go"
)

func runStmts(pass *analysis.Pass) (any, error) {
	if !inModule(pass) {
		return nil, nil
	}
	for _, f := range pass.Files {
		if isTest(pass, f.Pos()) {
			continue
		}
		ast.Inspect(f, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.BranchStmt:
				if x.Tok == token.GOTO {
					pass.Reportf(x.Pos(), "goto")
				}
			case *ast.GoStmt:
				if !atFile(pass, x.Pos(), modulePath, goFile) {
					pass.Reportf(x.Pos(), "go statement outside the root package's %s", goFile)
				}
			case *ast.CallExpr:
				if isBuiltin(pass, x.Fun, "recover") && !atFile(pass, x.Pos(), modulePath, recoverFile) {
					pass.Reportf(x.Pos(), "recover outside the root package's %s", recoverFile)
				}
			}
			return true
		})
	}
	return nil, nil
}

// isBuiltin reports whether e names the predeclared function name.
func isBuiltin(pass *analysis.Pass, e ast.Expr, name string) bool {
	id, ok := e.(*ast.Ident)
	if !ok || id.Name != name {
		return false
	}
	_, ok = pass.TypesInfo.Uses[id].(*types.Builtin)
	return ok
}
