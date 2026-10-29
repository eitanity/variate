// Package analyzers holds the go/analysis passes behind the custom checks:
// loop bound comments, no function-typed fields, no any or type switches in
// the engine, the statement bans, and a stated reason on every range over
// a map, whose order Go randomises. Package classification is by import
// path under the main module.
package analyzers
