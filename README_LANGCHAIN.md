# Tableau LangChain Starter Kit - LangChain Implementation

This project has been refactored to properly use LangChain instead of direct API calls, providing better maintainability, tool orchestration, and following LangChain best practices.

## 🏗️ **Architecture Overview**

### **Before (Direct API Approach):**
```
Frontend → FastAPI → Direct OpenAI API + Direct MCP HTTP calls
```

### **After (LangChain Approach):**
```
Frontend → FastAPI → LangChain Agent → LangChain Tools → MCP Server → Tableau
```

## 🔧 **Key Components**

### **1. LangChain Tools (`utilities/langchain_chat.py`)**
- **`list_tableau_workbooks`** - List available Tableau workbooks
- **`get_tableau_workbook`** - Get details of a specific workbook
- **`list_tableau_datasources`** - List available datasources
- **`query_tableau_datasource`** - Query datasources with custom queries
- **`get_tableau_view_image`** - Get dashboard/view images
- **`list_tableau_fields`** - List fields in a datasource

### **2. LangChain Agent**
- Uses `ChatOpenAI` with GPT-4o-mini
- Implements `create_openai_tools_agent` for tool orchestration
- Handles conversation history and context
- Automatic tool selection and execution

### **3. MCP Integration**
- Async MCP client for Tableau operations
- Streaming response handling
- Error handling and retry logic

## 🚀 **Benefits of LangChain Implementation**

### **1. Better Tool Management**
- **Centralized tool definitions** with proper schemas
- **Automatic tool selection** based on user queries
- **Built-in error handling** and retry logic

### **2. Improved Agent Behavior**
- **Conversation memory** with proper context management
- **Intelligent tool orchestration** - agents decide which tools to use
- **Better prompt engineering** with system messages

### **3. Enhanced Debugging**
- **LangSmith integration** for tracing and monitoring
- **Detailed logging** of tool calls and responses
- **Better error messages** and debugging information

### **4. Maintainability**
- **Modular design** - easy to add new tools
- **Type safety** with Pydantic models
- **Consistent patterns** following LangChain conventions

## 📁 **File Structure**

```
├── utilities/
│   └── langchain_chat.py          # LangChain implementation
├── web_app.py                     # Updated FastAPI app
├── web_app_langchain.py          # Standalone LangChain version
├── test_langchain.py             # LangChain tests
└── README_LANGCHAIN.md           # This file
```

## 🧪 **Testing**

### **Run LangChain Tests:**
```bash
source venv/bin/activate
python test_langchain.py
```

### **Test Results:**
- ✅ Basic chat functionality
- ✅ Tool calling and execution
- ✅ Data retrieval from Tableau
- ✅ Error handling

## 🔄 **Migration from Direct API**

### **What Changed:**
1. **Replaced direct OpenAI calls** with `ChatOpenAI`
2. **Replaced manual tool orchestration** with LangChain agents
3. **Added proper tool definitions** with schemas
4. **Improved error handling** and logging
5. **Added conversation memory** management

### **What Stayed the Same:**
1. **Frontend interface** - no changes needed
2. **MCP server integration** - same underlying calls
3. **API endpoints** - same interface
4. **Streaming responses** - same user experience

## 🛠️ **Usage Examples**

### **Basic Chat:**
```python
from utilities.langchain_chat import langchain_chat

result = await langchain_chat("What data sources do I have?")
print(result['response'])
```

### **Tool Execution:**
The agent automatically selects and executes tools based on user queries:
- "Show me the dashboard" → `get_tableau_view_image`
- "List my workbooks" → `list_tableau_workbooks`
- "Query the sales data" → `query_tableau_datasource`

## 🔧 **Configuration**

### **Environment Variables:**
```bash
OPENAI_API_KEY=your_openai_api_key
```

### **LangSmith Tracing (Optional):**
```python
# Automatically enabled if LANGCHAIN_API_KEY is set
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=tableau-langchain-starter
```

## 🚀 **Deployment**

### **Local Development:**
```bash
source venv/bin/activate
python web_app.py
```

### **Production:**
The LangChain implementation works with the same Vercel deployment:
- No changes to `vercel.json`
- Same environment variables
- Same static file serving

## 📊 **Performance Improvements**

### **Before:**
- Manual iteration loops
- Custom tool calling logic
- Basic error handling
- No conversation memory

### **After:**
- LangChain handles iterations automatically
- Intelligent tool selection
- Robust error handling and retries
- Proper conversation context management

## 🔍 **Debugging**

### **LangSmith Integration:**
- View tool calls and responses
- Monitor conversation flows
- Debug agent decision making
- Performance analytics

### **Console Logging:**
- Detailed tool execution logs
- Error tracking and reporting
- Performance metrics

## 🎯 **Next Steps**

1. **Add more tools** for specific Tableau operations
2. **Implement custom prompts** for different use cases
3. **Add conversation memory** persistence
4. **Implement tool result caching**
5. **Add more sophisticated error recovery**

## 🤝 **Contributing**

When adding new tools:
1. Define the tool function with proper type hints
2. Add it to the `TABLEAU_TOOLS` list
3. Update the system prompt if needed
4. Add tests for the new tool

The LangChain implementation provides a solid foundation for building more sophisticated Tableau AI applications while maintaining the simplicity and reliability of the original design.

