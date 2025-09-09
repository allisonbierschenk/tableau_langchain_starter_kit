#!/usr/bin/env python3
"""
Simple deployment test script to verify the app configuration
"""
import os
import sys

def test_environment_variables():
    """Test if required environment variables are accessible"""
    print("🔍 Testing environment variables...")
    
    # Check if we can access the environment variable
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✅ OPENAI_API_KEY is set")
        return True
    else:
        print("❌ OPENAI_API_KEY is not set")
        print("   This needs to be set in Vercel environment variables")
        return False

def test_configuration():
    """Test the app configuration"""
    print("\n🔍 Testing app configuration...")
    
    # Read the web_app.py file to check configuration
    try:
        with open('web_app.py', 'r') as f:
            content = f.read()
        
        # Check if CORS is properly configured
        if 'tableau-langchain-starter-kit.vercel.app' in content:
            print("✅ CORS configuration includes production URL")
        else:
            print("❌ CORS configuration missing production URL")
            return False
            
        # Check if MCP server URL is configured
        if 'MCP_SERVER_URL' in content:
            print("✅ MCP server URL is configured")
        else:
            print("❌ MCP server URL not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def test_frontend_config():
    """Test frontend configuration"""
    print("\n🔍 Testing frontend configuration...")
    
    try:
        with open('static/index.html', 'r') as f:
            content = f.read()
        
        # Check if production URL is set
        if 'https://tableau-langchain-starter-kit.vercel.app' in content:
            print("✅ Frontend is configured for production URL")
            return True
        else:
            print("❌ Frontend still using localhost URL")
            return False
            
    except Exception as e:
        print(f"❌ Error reading frontend config: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Tableau LangChain Starter Kit - Deployment Test")
    print("=" * 50)
    
    tests = [
        test_environment_variables,
        test_configuration,
        test_frontend_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your app should work for other users.")
        print("\n📋 Next steps:")
        print("1. Set OPENAI_API_KEY in Vercel environment variables")
        print("2. Deploy to Vercel")
        print("3. Test the deployed URL")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
