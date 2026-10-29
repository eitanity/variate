// Package dist holds the distribution families and their exact samplers.
// Every family implements Distribution; continuous families also implement
// Quantiler and discrete families PMFer. Simulation code never calls Fill
// directly: it draws through a Sampler.
package dist
