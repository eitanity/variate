package variate

import "testing"

// TestClaimAtMostOneBatchAhead: whatever slots are free, no task is
// claimed two batches ahead of the finaliser, whose batch's histograms
// it would otherwise write while they are merged.
func TestClaimAtMostOneBatchAhead(t *testing.T) {
	s := sched{active: true, batch: 4, tasks: 100, slotState: make([]int8, 16), slotTask: make([]int64, 16)}
	for fin := int64(0); fin < 20; fin++ {
		for next := fin; next < fin+12; next++ {
			s.finalNext, s.next = fin, next
			if got, want := s.canClaim(), next/4 <= fin/4+1; got != want {
				t.Errorf("finaliser at %d, next %d: claim %v", fin, next, got)
			}
		}
	}
}
