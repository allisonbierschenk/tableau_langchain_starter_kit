#!/usr/bin/env python3
"""
Simple test script for the Tableau AI Chat application
"""

import requests
import json
import time

def test_server():
    """Test the server endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Tableau AI Chat Server...")
    
    # Test health endpoint
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
    
    # Test datasources endpoint
    try:
        response = requests.get(f"{base_url}/datasources")
        if response.status_code == 200:
            data = response.json()
            print("✅ Datasources endpoint working")
            print(f"📊 Response: {data.get('response', 'No response')[:200]}...")
        else:
            print(f"❌ Datasources endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Datasources endpoint error: {e}")
    
    # Test chat endpoint
    try:
        chat_data = {"message": "What data sources do I have access to?"}
        response = requests.post(f"{base_url}/chat", json=chat_data)
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat endpoint working")
            print(f"💬 Response: {data.get('response', 'No response')[:200]}...")
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
    
    print("\n🎉 Test completed!")
    return True

if __name__ == "__main__":
    test_server()
