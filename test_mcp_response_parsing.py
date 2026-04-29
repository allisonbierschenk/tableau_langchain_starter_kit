"""
Test MCP response parsing - the actual structure from logs
"""

import sys
import json
sys.path.insert(0, '/Users/abierschenk/Desktop/TableauRepos/tableau_langchain_starter_kit')

from utilities.pulse_metric_enrich import _iter_json_roots_from_tool_result, extract_pulse_metric_id_to_name

# Simulate the ACTUAL tool result structure from logs
# The result is a STRING that contains a JSON object with content blocks
tool_result_as_string = json.dumps({
    "content": [
        {
            "type": "text",
            "text": json.dumps([
                {
                    "id": "63fefae0-755b-461a-8db6-0453f9bc7a6e",
                    "specification": {
                        "basic": {
                            "name": "Sales Performance"
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
            ])
        }
    ],
    "isError": False
})

tool_result = {
    "tool": "admin-pulse",
    "arguments": {"operation": "batch-get-metrics"},
    "result": tool_result_as_string  # This is how it comes from MCP
}

print("Testing MCP response parsing...")
print(f"Result type: {type(tool_result['result'])}")
print(f"Result preview: {tool_result['result'][:100]}...\n")

# Test root extraction
roots = _iter_json_roots_from_tool_result(tool_result)
print(f"\nExtracted {len(roots)} JSON roots")

# Test full extraction pipeline
pulse_map = extract_pulse_metric_id_to_name([tool_result])

print(f"\nFinal extraction:")
print(f"Found {len(pulse_map)} metric names")
for mid, mname in pulse_map.items():
    print(f"  ✓ {mid[:8]}... → {mname}")

assert len(pulse_map) == 2, f"Expected 2 metrics, got {len(pulse_map)}"
assert "63fefae0-755b-461a-8db6-0453f9bc7a6e" in pulse_map
assert pulse_map["63fefae0-755b-461a-8db6-0453f9bc7a6e"] == "Sales Performance"

print("\n✅ All tests passed! MCP response parsing works correctly")
