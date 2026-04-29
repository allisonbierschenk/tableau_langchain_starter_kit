"""
Test Pulse metric name extraction from actual batch-get-metrics response structure
"""

import sys
sys.path.insert(0, '/Users/abierschenk/Desktop/TableauRepos/tableau_langchain_starter_kit')

from utilities.pulse_metric_enrich import _walk_pulse_id_names

# Simulate the actual JSON structure from batch-get-metrics
sample_batch_response = [
    {
        "id": "63fefae0-755b-461a-8db6-0453f9bc7a6e",
        "specification": {
            "filters": [],
            "measurement_period": {
                "granularity": "GRANULARITY_BY_MONTH"
            },
            "basic": {
                "name": "Sales Performance",
                "description": "Monthly sales tracking"
            }
        }
    },
    {
        "id": "bd71f5a2-a7f5-4db8-bc6e-f581485dc8e9",
        "specification": {
            "basic": {
                "name": "Customer Growth"
            }
        }
    }
]

# Test extraction
pulse_map = {}
_walk_pulse_id_names(sample_batch_response, pulse_map)

print("Extraction test results:")
print(f"Found {len(pulse_map)} metric names")

for mid, mname in pulse_map.items():
    print(f"  ✓ {mid[:8]}... → {mname}")

# Verify we got both
assert len(pulse_map) == 2, f"Expected 2 metrics, got {len(pulse_map)}"
assert "63fefae0-755b-461a-8db6-0453f9bc7a6e" in pulse_map
assert "bd71f5a2-a7f5-4db8-bc6e-f581485dc8e9" in pulse_map
assert pulse_map["63fefae0-755b-461a-8db6-0453f9bc7a6e"] == "Sales Performance"
assert pulse_map["bd71f5a2-a7f5-4db8-bc6e-f581485dc8e9"] == "Customer Growth"

print("\n✅ All tests passed! Extraction pattern works for batch-get-metrics structure")
