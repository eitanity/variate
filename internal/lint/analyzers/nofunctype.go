package analyzers

import (
	"go/ast"

	"golang.org/x/tools/go/analysis"
)

// NoFuncType checks the engine packages: no struct field and no
// package-level variable has a function type.
var NoFuncType = &analysis.Analyzer{
	Name: "nofunctype",
	Doc:  "no function-typed fields or package variables in the root package, dist and stats",
	Run:  runNoFuncType,
}

func runNoFuncType(pass *analysis.Pass) (any, error) {
	if !inEngine(pass) {
		return nil, nil
	}
	for _, f := range pass.Files {
		if isTest(pass, f.Pos()) {
			continue
		}
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok {
				continue
			}
			for _, sp := range gd.Specs {
				checkSpec(pass, sp)
			}
		}
	}
	return nil, nil
}

// checkSpec reports function types in a var spec or in a struct type's fields.
func checkSpec(pass *analysis.Pass, sp ast.Spec) {
	switch s := sp.(type) {
	case *ast.ValueSpec:
		if _, ok := s.Type.(*ast.FuncType); ok {
			pass.Reportf(s.Pos(), "package-level variable of function type")
		}
	case *ast.TypeSpec:
		st, ok := s.Type.(*ast.StructType)
		if !ok {
			return
		}
		for _, fld := range st.Fields.List {
			if _, ok := fld.Type.(*ast.FuncType); ok {
				pass.Reportf(fld.Pos(), "function-typed struct field in %s", s.Name.Name)
			}
		}
	}
}
