// Package config loads a model file into a variate.Graph and a
// variate.Config: YAML with one family key per distribution, time units
// converted to periods through the file's period declaration, several
// graphs merged under their names, and templates expanded into
// independent copies. The file is untrusted input: it is capped in size,
// walked as a node tree with every unknown key an error, and any data
// file it names must sit beside it.
//
// A host that builds its model as data, not as a file, makes the
// variate.Graph itself and converts what a file would convert through
// the same functions the file reader uses: UnitScale for a time unit,
// BlockChain and BlockSource for a reliability block.
package config
