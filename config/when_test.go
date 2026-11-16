package config_test

import (
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// TestParseWhen: a condition at a value, at a return period, and a band
// read back as the engine names them; a malformed one is refused.
func TestParseWhen(t *testing.T) {
	for _, s := range []string{"total>=2e+06", "category:damage>=1/10", "total in [1/10,1/100)", "source:grid in [5,1/50)"} {
		c, err := config.ParseWhen(s)
		if err != nil {
			t.Fatalf("%s: %v", s, err)
		}
		if got := variate.WhenName(c); got != s {
			t.Errorf("%s read back as %s", s, got)
		}
	}
	for _, s := range []string{"total", "total>=x", "total>=1/x", "total in [1,2]", "total in 1,2"} {
		if _, err := config.ParseWhen(s); err == nil || !strings.Contains(err.Error(), "when "+s) {
			t.Errorf("%s: error %v", s, err)
		}
	}
}
