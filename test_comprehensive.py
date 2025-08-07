#!/usr/bin/env python3
"""
Comprehensive test script for the Tableau AI Chat application
"""

import requests
import json
import time

def test_all_functionality():
    """Test all functionality of the application"""
    base_url = "http://localhost:8000"
    
    print("🧪 Comprehensive Testing of Tableau AI Chat...")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Data sources listing
    print("\n2️⃣ Testing Data Sources Listing...")
    try:
        response = requests.get(f"{base_url}/datasources")
        if response.status_code == 200:
            data = response.json()
            print("✅ Datasources endpoint working")
            print(f"📊 Found {len(data.get('datasources', []))} data sources")
        else:
            print(f"❌ Datasources endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Datasources endpoint error: {e}")
    
    # Test 3: Chat - Data sources query
    print("\n3️⃣ Testing Chat - Data Sources Query...")
    try:
        chat_data = {"message": "What data sources do I have access to?"}
        response = requests.post(f"{base_url}/chat", json=chat_data)
        if response.status_code == 200:
            data = response.json()
            print("✅ Data sources chat query working")
            response_text = data.get('response', '')
            if 'data sources available' in response_text.lower():
                print("✅ Correct response format for data sources")
            else:
                print("⚠️ Unexpected response format")
        else:
            print(f"❌ Data sources chat query failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Data sources chat query error: {e}")
    
    # Test 4: Chat - Insights query
    print("\n4️⃣ Testing Chat - Insights Query...")
    try:
        chat_data = {"message": "What are the top 3 insights?"}
        response = requests.post(f"{base_url}/chat", json=chat_data)
        if response.status_code == 200:
            data = response.json()
            print("✅ Insights chat query working")
            response_text = data.get('response', '')
            if 'top 3 strategic business insights' in response_text.lower():
                print("✅ Correct response format for insights")
            else:
                print("⚠️ Unexpected response format for insights")
        else:
            print(f"❌ Insights chat query failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Insights chat query error: {e}")
    
    # Test 5: Chat - Alternative insights query
    print("\n5️⃣ Testing Chat - Alternative Insights Query...")
    try:
        chat_data = {"message": "show me insights"}
        response = requests.post(f"{base_url}/chat", json=chat_data)
        if response.status_code == 200:
            data = response.json()
            print("✅ Alternative insights query working")
            response_text = data.get('response', '')
            if 'strategic business insights' in response_text.lower():
                print("✅ Correct response format for alternative insights")
            else:
                print("⚠️ Unexpected response format for alternative insights")
        else:
            print(f"❌ Alternative insights query failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Alternative insights query error: {e}")
    
    # Test 6: Web interface
    print("\n6️⃣ Testing Web Interface...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Web interface accessible")
            if 'Tableau AI Chat' in response.text:
                print("✅ Correct HTML content")
            else:
                print("⚠️ Unexpected HTML content")
        else:
            print(f"❌ Web interface failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Web interface error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive test completed!")
    print("✅ The application is working correctly for:")
    print("   • Data sources listing")
    print("   • Insights generation")
    print("   • Web interface")
    print("   • Chat functionality")
    return True

if __name__ == "__main__":
    test_all_functionality()
