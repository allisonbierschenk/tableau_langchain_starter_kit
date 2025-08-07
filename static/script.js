// script.js - STANDALONE VERSION WITHOUT TABLEAU EXTENSIONS API

let currentStream = null;
let conversationHistory = []; // Track conversation for MCP context
let sessionId = null; // Track session ID for better client-server communication

// Generate a session ID for this client
function generateSessionId() {
    if (!sessionId) {
        sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    return sessionId;
}

// --- Enhanced Message Display Helper ---
function addMessage(html, type, messageId = null) {
    const chatBox = document.getElementById('chatBox');
    if (!chatBox) return;

    let messageDiv;
    if (messageId && document.getElementById(messageId)) {
        messageDiv = document.getElementById(messageId);
    } else {
        messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        if (messageId) messageDiv.id = messageId;
        chatBox.appendChild(messageDiv);
    }
    
    // Enhanced HTML rendering with better formatting
    messageDiv.innerHTML = html;
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}

// --- Enhanced Progress Indicator ---
function showTypingIndicator(show = true) {
    const existingIndicator = document.getElementById('typing-indicator');
    if (show && !existingIndicator) {
        const indicatorHtml = `
            <div class="typing-indicator" id="typing-indicator">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
                <span class="typing-text">AI is analyzing your data...</span>
            </div>
        `;
        addMessage(indicatorHtml, 'bot', 'typing-indicator');
    } else if (!show && existingIndicator) {
        existingIndicator.remove();
    }
}

// --- Improved MCP Detection ---
function shouldUseMCP(message) {
    const mcpKeywords = [
        'insight', 'insights', 'metric', 'metrics', 'kpi', 'pulse', 'dashboard',
        'analytics', 'performance', 'trend', 'trends', 'analysis', 'visualiz',
        'chart', 'graph', 'plot', 'business intelligence', 'bi', 'top', 'best',
        'worst', 'focus', 'proactive', 'recommend', 'should', 'opportunity',
        'risk', 'pattern', 'anomaly', 'compare', 'comparison', 'overview',
        'summary', 'key', 'important', 'critical', 'highlight'
    ];
    
    const lowerMessage = message.toLowerCase();
    
    // Also trigger MCP for questions that sound analytical
    const analyticalPatterns = [
        /what.*should.*focus/i,
        /what.*insights?/i,
        /tell me about/i,
        /show me.*top/i,
        /what.*important/i,
        /what.*key/i,
        /give me.*analysis/i,
        /help me understand/i
    ];
    
    return mcpKeywords.some(keyword => lowerMessage.includes(keyword)) ||
           analyticalPatterns.some(pattern => pattern.test(message));
}

// --- Enhanced Streaming Chat with Better Error Handling ---
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) {
        input.focus();
        return;
    }

    // Prevent sending multiple messages simultaneously
    if (currentStream) {
        addMessage("⏳ Please wait for the current response to complete.", "bot");
        return;
    }

    // Add user message to conversation history
    conversationHistory.push({ role: 'user', content: message });

    // Display user message
    addMessage(escapeHtml(message), 'user');
    input.value = '';
    
    // Update UI state
    const sendBtn = document.getElementById('sendBtn');
    input.disabled = true;
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<span class="spinner"></span> Analyzing...';

    const botMessageId = 'bot-response-' + Date.now();
    let fullResponse = '';
    let hasStartedResponse = false;
    let usedMCP = false;

    try {
        const controller = new AbortController();
        currentStream = controller;

        // Determine which endpoint to use
        usedMCP = shouldUseMCP(message);
        console.log(`🔄 Using ${usedMCP ? 'MCP streaming' : 'traditional'} endpoint for: "${message}"`);

        if (usedMCP) {
            // Try MCP first
            try {
                showTypingIndicator();
                fullResponse = await handleMCPRequest(message, controller, botMessageId);
                hasStartedResponse = true;
            } catch (mcpError) {
                console.warn('MCP failed, falling back to traditional agent:', mcpError);
                showTypingIndicator(false);
                addMessage("🔄 <em>Switching to traditional analysis...</em>", "bot");
                
                // Fallback to traditional endpoint
                fullResponse = await handleTraditionalRequest(message, controller);
                hasStartedResponse = true;
            }
        } else {
            // Use traditional endpoint directly
            showTypingIndicator();
            fullResponse = await handleTraditionalRequest(message, controller);
            hasStartedResponse = true;
        }

        // Display final response if we have one
        if (fullResponse && hasStartedResponse) {
            showTypingIndicator(false);
            addMessage(formatResponse(fullResponse), 'bot', botMessageId);
            conversationHistory.push({ role: 'assistant', content: fullResponse });
        } else if (!hasStartedResponse) {
            // If nothing worked, show error
            showTypingIndicator(false);
            addMessage("❌ I apologize, but I encountered an issue analyzing your request. Please try rephrasing your question or asking something more specific about your data.", 'bot', botMessageId);
        }

    } catch (error) {
        showTypingIndicator(false);
        console.error('Chat error:', error);
        
        let errorMessage = error.message;
        if (error.name === 'AbortError') {
            errorMessage = 'Request was cancelled.';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Unable to connect to the server. Please check your connection and try again.';
        }
        
        addMessage(`❌ <strong>Error:</strong><br>${escapeHtml(errorMessage)}<br><br>💡 <em>Try asking a more specific question about your data.</em>`, 'bot', botMessageId);
    } finally {
        // Reset UI state
        currentStream = null;
        input.disabled = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = 'Send';
        input.focus();
    }
}

// --- Handle MCP Request ---
async function handleMCPRequest(message, controller, botMessageId) {
    const response = await fetch('/mcp-chat-stream', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            'X-Session-ID': generateSessionId()
        },
        body: JSON.stringify({
            messages: conversationHistory.slice(-10),
            query: message
        }),
        signal: controller.signal
    });

    if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorMessage;
        } catch (e) {
            // If we can't parse JSON, use the status text
        }
        throw new Error(errorMessage);
    }

    return await handleMCPStreamingResponse(response, botMessageId);
}

// --- Handle Traditional Request ---
async function handleTraditionalRequest(message, controller) {
    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'X-Session-ID': generateSessionId()
        },
        body: JSON.stringify({ message }),
        signal: controller.signal
    });

    if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorMessage;
        } catch (e) {
            // If we can't parse JSON, use the status text
        }
        throw new Error(errorMessage);
    }

    const data = await response.json();
    return data.response || 'No response received';
}

// --- Enhanced MCP Streaming Response Handler ---
async function handleMCPStreamingResponse(response, botMessageId) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    let progressDiv = null;

    // Remove typing indicator when we start receiving real data
    showTypingIndicator(false);
    
    // Create progress indicator for MCP
    const progressId = 'mcp-progress-' + Date.now();
    progressDiv = addMessage('<div class="mcp-progress">🔄 <em>Initializing advanced analysis...</em></div>', 'bot', progressId);
    
    let botMessageDiv = addMessage('', 'bot', botMessageId);

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split('\n\n');
            buffer = events.pop() || ''; // Keep incomplete event in buffer

            for (const eventBlock of events) {
                if (!eventBlock.trim()) continue;

                try {
                    const lines = eventBlock.split('\n');
                    let eventType = null;
                    let eventData = null;

                    for (const line of lines) {
                        if (line.startsWith('event: ')) {
                            eventType = line.substring(7).trim();
                        } else if (line.startsWith('data: ')) {
                            const dataStr = line.substring(6).trim();
                            if (dataStr) {
                                try {
                                    eventData = JSON.parse(dataStr);
                                } catch (parseError) {
                                    console.warn('Failed to parse event data:', dataStr, parseError);
                                    continue;
                                }
                            }
                        }
                    }

                    if (eventType && eventData) {
                        if (eventType === 'progress' && progressDiv) {
                            handleMCPProgressEvent(eventData, progressDiv);
                        } else if (eventType === 'result') {
                            fullResponse = eventData.response || '';
                            if (fullResponse) {
                                // Enhanced formatting for MCP results
                                botMessageDiv.innerHTML = formatMCPResponse(fullResponse);
                            }
                            // Remove progress indicator
                            if (progressDiv) {
                                progressDiv.remove();
                                progressDiv = null;
                            }
                            break;
                        } else if (eventType === 'error') {
                            throw new Error(eventData.error || 'Unknown MCP stream error');
                        } else if (eventType === 'done') {
                            if (progressDiv) {
                                progressDiv.remove();
                            }
                            break;
                        }
                    }
                } catch (parseError) {
                    console.warn('Failed to parse MCP event:', eventBlock, parseError);
                    continue;
                }
            }
        }

        // Ensure we have some response
        if (!fullResponse) {
            throw new Error('No response received from MCP analysis');
        }

        return fullResponse;

    } finally {
        if (progressDiv) {
            progressDiv.remove();
        }
    }
}

// --- Handle MCP Progress Events ---
function handleMCPProgressEvent(eventData, progressDiv) {
    if (eventData.message) {
        const icon = getMCPProgressIcon(eventData.step);
        const iteration = eventData.iteration ? ` (Step ${eventData.iteration}/${eventData.maxIterations || 'N/A'})` : '';
        progressDiv.innerHTML = `<div class="mcp-progress">${icon} <em>${escapeHtml(eventData.message)}${iteration}</em></div>`;
    }
}

// --- Get Progress Icon for MCP Steps ---
function getMCPProgressIcon(step) {
    const icons = {
        'init': '🔌',
        'tools': '🔍',
        'tools-found': '🛠️',
        'analysis-start': '🚀',
        'iteration-start': '🔄',
        'tools-executing': '⚙️',
        'tool-executing': '🔧',
        'tool-completed': '✅',
        'tool-error': '⚠️',
        'iteration-complete': '✨',
        'complete': '🎉',
        'max-iterations': '⏰'
    };
    return icons[step] || '📋';
}

// --- Enhanced Response Formatting for MCP Results ---
function formatMCPResponse(text) {
    if (!text) return '';
    
    let formatted = text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // **bold**
        .replace(/\*(.*?)\*/g, '<em>$1</em>') // *italic*
        .replace(/`(.*?)`/g, '<code>$1</code>') // `code`
        .replace(/#{1,6}\s+(.*?)(?=\n|$)/g, '<strong>$1</strong>'); // # headers

    // Enhanced formatting for data tables and insights
    formatted = formatDataTables(formatted);
    formatted = formatInsightBoxes(formatted);
    
    return formatted;
}

// --- Format Insight Boxes ---
function formatInsightBoxes(text) {
    // Look for insight patterns and highlight them
    text = text.replace(/💡\s*(.*?)(?=\n|$)/g, '<div class="insight-box">💡 <strong>$1</strong></div>');
    text = text.replace(/🚨\s*(.*?)(?=\n|$)/g, '<div class="alert-box">🚨 <strong>$1</strong></div>');
    text = text.replace(/📈\s*(.*?)(?=\n|$)/g, '<div class="trend-box">📈 <strong>$1</strong></div>');
    text = text.replace(/🎯\s*(.*?)(?=\n|$)/g, '<div class="action-box">🎯 <strong>$1</strong></div>');
    
    return text;
}

// --- Format Data Tables in Response ---
function formatDataTables(text) {
    // Look for table-like patterns and enhance them
    const tableRegex = /(\|.+\|\s*\n\|[-:\s|]+\|\s*\n(?:\|.+\|\s*\n?)*)/g;
    
    return text.replace(tableRegex, (match) => {
        const lines = match.trim().split('\n').filter(line => line.trim());
        if (lines.length < 3) return match;
        
        try {
            // Parse header
            const headerCells = lines[0].split('|')
                .map(cell => cell.trim())
                .filter(cell => cell.length > 0);
            
            // Parse data rows (skip separator line)
            const dataRows = lines.slice(2)
                .filter(line => line.includes('|'))
                .map(line => 
                    line.split('|')
                        .map(cell => cell.trim())
                        .filter((cell, index, arr) => {
                            return !(cell === '' && (index === 0 || index === arr.length - 1));
                        })
                );
            
            if (headerCells.length === 0 || dataRows.length === 0) {
                return match;
            }
            
            // Generate HTML table
            let tableHtml = '<div class="data-table-wrapper"><table class="data-table">';
            tableHtml += '<thead><tr>';
            headerCells.forEach(header => {
                tableHtml += `<th>${escapeHtml(header)}</th>`;
            });
            tableHtml += '</tr></thead><tbody>';
            
            dataRows.forEach(row => {
                tableHtml += '<tr>';
                row.forEach((cell, index) => {
                    if (index < headerCells.length) {
                        tableHtml += `<td>${escapeHtml(cell)}</td>`;
                    }
                });
                tableHtml += '</tr>';
            });
            
            tableHtml += '</tbody></table></div>';
            return tableHtml;
            
        } catch (e) {
            console.warn('Error formatting table:', e);
            return match;
        }
    });
}

// --- Standard Response Formatting Helper ---
function formatResponse(text) {
    if (!text) return '';
    
    let formatted = text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // **bold**
        .replace(/\*(.*?)\*/g, '<em>$1</em>') // *italic*
        .replace(/`(.*?)`/g, '<code>$1</code>') // `code`
        .replace(/#{1,6}\s+(.*?)(?=\n|$)/g, '<strong>$1</strong>'); // # headers

    // Add insight formatting for regular responses too
    formatted = formatInsightBoxes(formatted);
    formatted = formatDataTables(formatted);
    
    return formatted;
}

// --- HTML Escaping Helper ---
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// --- Enhanced Enter Key Handler ---
function handleEnter(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    } else if (event.key === 'Escape') {
        // Cancel current stream if running
        if (currentStream) {
            currentStream.abort();
            currentStream = null;
        }
    }
}

// --- Enhanced Keyboard Shortcuts ---
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Ctrl/Cmd + Enter to send message
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            event.preventDefault();
            sendMessage();
        }
        
        // Escape to cancel current request
        if (event.key === 'Escape' && currentStream) {
            currentStream.abort();
            currentStream = null;
            addMessage("⏹️ <em>Request cancelled by user</em>", "bot");
        }
    });
}

// --- Enhanced Connection Health Check ---
async function checkServerHealth() {
    try {
        const response = await fetch('/health', { 
            method: 'GET',
            timeout: 5000 
        });
        return response.ok;
    } catch (error) {
        console.warn('Server health check failed:', error);
        return false;
    }
}

// --- Enhanced DOMContentLoaded Event Listener ---
document.addEventListener('DOMContentLoaded', async function() {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    // Setup event listeners
    if (messageInput) messageInput.addEventListener('keypress', handleEnter);
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);
    
    setupKeyboardShortcuts();

    // Auto-focus on input
    if (messageInput) {
        setTimeout(() => messageInput.focus(), 100);
    }

    // Check server health
    const serverHealthy = await checkServerHealth();
    if (!serverHealthy) {
        addMessage("⚠️ <strong>Server Connection Issue</strong><br>Unable to connect to the backend server. Please ensure the Python server is running and accessible.", "bot");
        return;
    }

    // Show ready message
    addMessage("✅ <strong>Ready!</strong> You can now ask me about your Tableau data sources and get insights.", "bot");
});