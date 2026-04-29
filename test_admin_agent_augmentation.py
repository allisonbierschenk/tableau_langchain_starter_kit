"""
Test script to verify admin agent context augmentation fixes
Tests the scenario from Slack where "viewers" response should trigger add-user, not list-users
"""

import sys
sys.path.insert(0, '/Users/abierschenk/Desktop/TableauRepos/tableau_langchain_starter_kit')

from utilities.admin_agent import _augment_query_with_context

def test_multi_email_role_followup():
    """Test that when user responds with role after being asked, it augments correctly"""

    # Simulate conversation where bot asked for site role for 2 emails
    conversation_history = [
        {
            "role": "user",
            "content": "add Abanana@Agmail.com and AAbanana@Agmail.com to the site"
        },
        {
            "role": "assistant",
            "content": "To add the users to the site, I need to know the site role you would like to assign to each of them. Could you please specify the site roles for the following emails?\n\n1. Abanana@Agmail.com\n2. AAbanana@Agmail.com"
        }
    ]

    # User responds with just "viewers"
    query = "viewers"

    augmented = _augment_query_with_context(query, conversation_history)

    print(f"Original query: '{query}'")
    print(f"Augmented query: '{augmented}'")

    # Check that augmentation is explicit about ADDING users
    assert "add" in augmented.lower(), f"Augmented query should contain 'add': {augmented}"
    assert "abanana@agmail.com" in augmented.lower(), f"Should include first email: {augmented}"
    assert "aabanana@agmail.com" in augmented.lower(), f"Should include second email: {augmented}"
    assert "viewers" in augmented.lower() or "viewer" in augmented.lower(), f"Should include role: {augmented}"

    # Most importantly, ensure it says "to the site" not "with site role" (ambiguous)
    assert "to the site" in augmented.lower(), f"Should be explicit about adding TO THE SITE: {augmented}"

    print("✅ Test passed! Augmentation creates unambiguous add-user instruction")

def test_single_email_role_followup():
    """Test single email case"""

    conversation_history = [
        {
            "role": "user",
            "content": "add john@example.com to the site"
        },
        {
            "role": "assistant",
            "content": "What site role should be assigned to john@example.com?"
        }
    ]

    query = "creator"
    augmented = _augment_query_with_context(query, conversation_history)

    print(f"\nSingle email test:")
    print(f"Original: '{query}'")
    print(f"Augmented: '{augmented}'")

    assert "add" in augmented.lower()
    assert "john@example.com" in augmented.lower()
    assert "creator" in augmented.lower()
    assert "to the site" in augmented.lower()

    print("✅ Single email test passed!")

if __name__ == "__main__":
    test_multi_email_role_followup()
    test_single_email_role_followup()
    print("\n🎉 All tests passed!")
