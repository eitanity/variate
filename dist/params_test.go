package dist

import (
	"math"
	"testing"
)

func near(a, b, rel float64) bool { return math.Abs(a-b) <= rel*math.Abs(b) }

func TestParameterisations(t *testing.T) {
	if d, err := NewLognormalFromMeanSD(50, 20); err != nil || !near(d.Mean(), 50, 1e-12) || !near(math.Sqrt(d.Var()), 20, 1e-12) {
		t.Error("lognormal from mean, sd", err)
	}
	if d, err := NewLognormalFromMedianP90(4, 24); err != nil || !near(d.Quantile(0.5), 4, 1e-12) || !near(d.Quantile(0.9), 24, 1e-12) {
		t.Error("lognormal from median, p90", err)
	}
	if d, err := NewGammaFromMeanSD(7, 2); err != nil || !near(d.Mean(), 7, 1e-12) || !near(math.Sqrt(d.Var()), 2, 1e-12) {
		t.Error("gamma from mean, sd", err)
	}
	if d, err := NewGammaFromShapeRate(2, 4); err != nil || !near(d.Mean(), 0.5, 1e-12) {
		t.Error("gamma from shape, rate", err)
	}
	if d, err := NewBetaFromMeanSD(0.3, 0.1); err != nil || !near(d.Mean(), 0.3, 1e-12) || !near(math.Sqrt(d.Var()), 0.1, 1e-12) {
		t.Error("beta from mean, sd", err)
	}
	if d, err := NewExponentialFromRate(4); err != nil || !near(d.Mean(), 0.25, 1e-12) {
		t.Error("exponential from rate", err)
	}
	if _, err := NewBetaFromMeanSD(0.5, 0.5); err == nil {
		t.Error("beta accepted sd^2 >= mean(1-mean)")
	}
	if _, err := NewLognormalFromMedianP90(4, 3); err == nil {
		t.Error("lognormal accepted p90 < median")
	}
}
