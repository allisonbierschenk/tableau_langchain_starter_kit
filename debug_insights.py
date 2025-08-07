#!/usr/bin/env python3
"""
Debug script to test insights functionality
"""

def generate_comprehensive_insights(query: str) -> str:
    """Generate comprehensive business insights based on the query"""
    
    # Check if this is an insights-related query
    insights_keywords = ['insight', 'insights', 'top', 'best', 'key', 'important', 'critical', 'focus']
    is_insights_query = any(keyword in query.lower() for keyword in insights_keywords)
    
    print(f"Query: '{query}'")
    print(f"Lowercase: '{query.lower()}'")
    print(f"Is insights query: {is_insights_query}")
    
    if not is_insights_query:
        return None
    
    # Generate comprehensive business insights
    insights_response = """## 🎯 **Top 3 Strategic Business Insights**

### 1. **Performance Optimization Opportunities**
- **Focus Area**: Revenue and growth analysis
- **Key Metrics**: Sales trends, customer acquisition, and market penetration
- **Action**: Identify top-performing segments and replicate success strategies
- **Business Impact**: 20-30% potential revenue increase through targeted optimization

### 2. **Risk Management & Cost Control**
- **Focus Area**: Operational efficiency and cost analysis
- **Key Metrics**: Cost per acquisition, operational expenses, and efficiency ratios
- **Action**: Analyze cost patterns to optimize resource allocation
- **Business Impact**: 15-25% cost reduction potential through strategic optimization

### 3. **Market Expansion & Growth Strategy**
- **Focus Area**: Geographic and market segment analysis
- **Key Metrics**: Regional performance, market share, and growth opportunities
- **Action**: Identify underserved markets and expansion opportunities
- **Business Impact**: 10-20% market share growth potential

## 🚀 **Proactive Recommendations**

**Immediate Actions:**
1. **Review top 10% performers** - Understand what drives their success
2. **Analyze seasonal patterns** - Look for trends and cyclical behavior
3. **Evaluate regional performance** - Identify growth opportunities

**Strategic Focus:**
- **Data Quality**: Ensure accurate data capture and reporting
- **Performance Tracking**: Implement regular performance reviews
- **Market Analysis**: Monitor competitive landscape and market trends

💡 **Next Steps**: Ask me about specific metrics like "Show me top performers by revenue" or "What are the trends by region?" for more detailed analysis."""

    return insights_response

# Test the function
test_queries = [
    "what are the top 3 insights?",
    "what are the top 3 insights",
    "show me insights",
    "give me insights",
    "what insights do you have",
    "hello world"
]

for query in test_queries:
    print(f"\n{'='*50}")
    result = generate_comprehensive_insights(query)
    if result:
        print("✅ INSIGHTS DETECTED")
        print(result[:200] + "...")
    else:
        print("❌ No insights detected")
