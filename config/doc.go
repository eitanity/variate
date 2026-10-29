// Package config loads a model file into a variate.Graph and a
// variate.Config: YAML with one family key per distribution, time units
// converted to periods through the file's period declaration, several
// graphs merged under their names, and templates expanded into
// independent copies. The file is untrusted input: it is capped in size,
// walked as a node tree with every unknown key an error, and any data
// file it names must sit beside it.
package config
