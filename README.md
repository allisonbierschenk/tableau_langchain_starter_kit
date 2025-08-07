# Tableau AI Chat

A standalone web application that provides intelligent chat interface for Tableau data sources and business insights.

## Features

### 🎯 **Data Sources Discovery**
- Lists all available data sources from your Tableau Server
- Provides detailed information including project, ID, and description
- Accessible via API or chat interface

### 💡 **Intelligent Insights Generation**
- Provides comprehensive business insights and recommendations
- Responds to queries like "What are the top 3 insights?" or "Show me insights"
- Offers strategic business analysis with actionable recommendations

### 🌐 **Web Interface**
- Clean, modern chat interface
- Real-time responses
- Mobile-responsive design
- No Tableau Extensions API dependency

## Quick Start

### 1. Setup Environment Variables
Create a `.env` file with your Tableau credentials:

```bash
TABLEAU_USER=your_username
TABLEAU_DOMAIN_FULL=https://your-tableau-server.com
TABLEAU_SITE=your_site_name
TABLEAU_JWT_CLIENT_ID=your_jwt_client_id
TABLEAU_JWT_SECRET_ID=your_jwt_secret_id
TABLEAU_JWT_SECRET=your_jwt_secret
TABLEAU_API_VERSION=3.21
OPENAI_API_KEY=your_openai_api_key
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python web_app.py
```

### 4. Access the Application
Open your browser and go to: `http://localhost:8000`

## Usage Examples

### Data Sources Queries
- "What data sources do I have access to?"
- "List my available data sources"
- "Show me all datasources"

### Insights Queries
- "What are the top 3 insights?"
- "Show me insights"
- "Give me business insights"
- "What insights do you have?"

## API Endpoints

### GET `/health`
Health check endpoint

### GET `/datasources`
Returns all available data sources from Tableau Server

### POST `/chat`
Chat endpoint that handles both data sources and insights queries

### GET `/`
Web interface

## Architecture

The application has been simplified to work as a standalone service without requiring the Tableau Extensions API. Key components:

- **FastAPI Backend**: Handles API requests and chat functionality
- **Tableau REST API Integration**: Fetches data sources directly from Tableau Server
- **Insights Engine**: Provides comprehensive business insights and recommendations
- **Modern Web Interface**: Clean, responsive chat interface

## Testing

Run the comprehensive test suite:

```bash
python test_comprehensive.py
```

This will test:
- Health check functionality
- Data sources listing
- Insights generation
- Web interface accessibility
- Chat functionality

## Key Improvements

1. **Removed Tableau Extensions API dependency** - Now works as a standalone application
2. **Added intelligent insights generation** - Responds to business insight queries
3. **Enhanced web interface** - Modern, responsive design
4. **Comprehensive testing** - Full test suite for all functionality
5. **Better error handling** - Robust error handling and fallback responses

## Example Responses

### Data Sources Query
```
Here are the data sources available:

Name: Superstore Datasource
Project: Samples
ID: 967b0f71-6192-41c4-9434-13302397b031

Name: TS Users
Project: Admin Insights
ID: c79a393e-ac72-48c1-8338-3d802d7973d9
...
```

### Insights Query
```
## 🎯 **Top 3 Strategic Business Insights**

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

💡 **Next Steps**: Ask me about specific metrics like "Show me top performers by revenue" or "What are the trends by region?" for more detailed analysis.
```
