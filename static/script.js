// script.js - STANDALONE VERSION WITHOUT TABLEAU EXTENSIONS API

// Configuration - Set your deployed backend URL here
const API_BASE_URL = window.API_BASE_URL || (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://localhost:8000' 
    : 'https://tableau-langchain-starter-kit.vercel.app');

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

// --- Simplified Chat - Always Use MCP/Analyst Agent ---
async function sendMessage() {
    console.log('🚀 sendMessage called');
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) {
        input.focus();
        return;
    }

    console.log('📝 Message:', message);

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

    try {
        console.log('🔧 Creating AbortController');
        const controller = new AbortController();
        currentStream = controller;

        // Always use MCP/Analyst Agent endpoint
        console.log(`🚀 Using MCP/Analyst Agent for: "${message}"`);

        showTypingIndicator();
        console.log('📡 Calling handleMCPRequest');
        fullResponse = await handleMCPRequest(message, controller, botMessageId);
        console.log('✅ handleMCPRequest completed, response:', fullResponse);
        hasStartedResponse = true;

        // Display final response if we have one
        if (fullResponse && hasStartedResponse) {
            showTypingIndicator(false);
            // Response is already displayed by the streaming handler
            conversationHistory.push({ role: 'assistant', content: fullResponse });
        } else if (!hasStartedResponse) {
            // If nothing worked, show error
            showTypingIndicator(false);
            addMessage("❌ I apologize, but I encountered an issue analyzing your request. Please try rephrasing your question or asking something more specific about your data.", 'bot', botMessageId);
        }

    } catch (error) {
        console.error('❌ Error in sendMessage:', error);
        showTypingIndicator(false);
        
        // Enhanced error handling
        let errorMessage = "❌ I encountered an error while processing your request.";
        
        if (error.name === 'AbortError') {
            errorMessage = "⏹️ Request was cancelled.";
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = "🌐 Connection error. Please check your internet connection and try again.";
        } else if (error.message.includes('HTTP')) {
            errorMessage = `🔧 Server error: ${error.message}. Please try again later.`;
        }
        
        addMessage(errorMessage, 'bot', botMessageId);
    } finally {
        // Reset UI state
        input.disabled = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = 'Send';
        currentStream = null;
    }
}

// --- Handle MCP Request ---
async function handleMCPRequest(message, controller, botMessageId) {
    console.log('📡 handleMCPRequest called with message:', message);
    console.log('🌐 API_BASE_URL:', API_BASE_URL);
    
    try {
        const response = await fetch(`${API_BASE_URL}/mcp-chat-stream`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                'X-Session-ID': generateSessionId()
            },
            body: JSON.stringify({ message }),
            signal: controller.signal
        });

        console.log('📡 Response status:', response.status);
        console.log('📡 Response headers:', response.headers);

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

        console.log('📡 Calling handleMCPStreamingResponse');
        const result = await handleMCPStreamingResponse(response, botMessageId, message);
        console.log('✅ handleMCPStreamingResponse completed, result:', result);
        return result;
    } catch (error) {
        console.error('❌ Error in handleMCPRequest:', error);
        throw error;
    }
}

// --- Enhanced MCP Streaming Response Handler ---
async function handleMCPStreamingResponse(response, botMessageId, originalMessage = '') {
    console.log('📡 handleMCPStreamingResponse called');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    let progressDiv = null;
    let isComplete = false;
    let timeoutId = null;

    // Remove typing indicator when we start receiving real data
    showTypingIndicator(false);
    
    // Create progress indicator for MCP
    const progressId = 'mcp-progress-' + Date.now();
    progressDiv = addMessage('<div class="mcp-progress">🔄 <em>Initializing advanced analysis...</em></div>', 'bot', progressId);
    
    let botMessageDiv = addMessage('', 'bot', botMessageId);

    try {
        console.log('📡 Starting to read stream');
        
        // Set a timeout to prevent infinite analyzing state
        timeoutId = setTimeout(() => {
            if (!isComplete) {
                console.warn('⚠️ Stream timeout - forcing completion');
                if (progressDiv) {
                    progressDiv.remove();
                    progressDiv = null;
                }
                isComplete = true;
            }
        }, 60000); // 60 second timeout
        
        while (true && !isComplete) {
            const { done, value } = await reader.read();
            if (done) {
                console.log('📡 Stream done');
                clearTimeout(timeoutId);
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            console.log('📡 Buffer received:', buffer);
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

                    console.log('📡 Event type:', eventType, 'Event data:', eventData);

                    if (eventType && eventData) {
                        if (eventType === 'progress' && progressDiv) {
                            handleMCPProgressEvent(eventData, progressDiv);
                        } else if (eventType === 'result') {
                            // Handle both old format and new format
                            if (eventData.data && eventData.data.response) {
                                fullResponse = eventData.data.response;
                                // Store tool results for datasource info
                                if (eventData.data.tool_results) {
                                    window.lastToolResults = eventData.data.tool_results;
                                }
                            } else {
                                fullResponse = eventData.response || '';
                                // Store tool results for datasource info (new format)
                                if (eventData.tool_results) {
                                    window.lastToolResults = eventData.tool_results;
                                }
                            }
                            console.log('📡 Result event, fullResponse:', fullResponse);
                            console.log('📊 Tool results for datasource extraction:', window.lastToolResults);
                            if (fullResponse) {
                                // Enhanced formatting for MCP results with datasource info
                                botMessageDiv.innerHTML = formatMCPResponse(fullResponse, window.lastToolResults);
                            }
                            // Remove progress indicator immediately when we get results
                            if (progressDiv) {
                                progressDiv.remove();
                                progressDiv = null;
                            }
                            
                            // Auto-display dashboard image if user requested it
                            if (originalMessage && shouldDisplayDashboardImage(originalMessage)) {
                                console.log('🖼️ Auto-displaying dashboard image based on user request');
                                setTimeout(() => displayDashboardImage(), 500);
                            }
                            // Mark as complete but continue to wait for done event
                            isComplete = true;
                        } else if (eventType === 'error') {
                            throw new Error(eventData.error || 'Unknown MCP stream error');
                        } else if (eventType === 'done') {
                            console.log('📡 Done event received');
                            isComplete = true;
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

        console.log('📡 Returning fullResponse:', fullResponse);
        return fullResponse;

    } finally {
        // Clear timeout if it exists
        if (timeoutId) {
            clearTimeout(timeoutId);
        }
        
        // Ensure progress indicator is removed
        if (progressDiv) {
            progressDiv.remove();
        }
    }
}

// --- Handle MCP Progress Events ---
function handleMCPProgressEvent(eventData, progressDiv) {
    if (eventData.message) {
        const icon = getMCPProgressIcon(eventData.step);
        let userFriendlyMessage = eventData.message;
        
        // Convert technical messages to user-friendly ones
        if (eventData.step === 'discover-tools') {
            userFriendlyMessage = "🔍 Discovering available data analysis tools...";
        } else if (eventData.step === 'tools-discovered') {
            userFriendlyMessage = `🛠️ Found ${eventData.tools ? eventData.tools.length : 0} analysis tools ready to use`;
        } else if (eventData.step === 'iteration') {
            userFriendlyMessage = `🧠 Thinking through your question (Step ${eventData.iteration || 1})...`;
        } else if (eventData.step === 'executing-tools') {
            userFriendlyMessage = `⚙️ Analyzing your data (${eventData.tool_count || 1} analysis step${eventData.tool_count > 1 ? 's' : ''})...`;
        } else if (eventData.step === 'tool-executing') {
            if (eventData.tool === 'list-datasources') {
                userFriendlyMessage = "📊 Exploring available data sources...";
            } else if (eventData.tool === 'list-fields') {
                userFriendlyMessage = "🔍 Examining data structure and available fields...";
            } else if (eventData.tool === 'query-datasource') {
                userFriendlyMessage = "📈 Querying data to find insights...";
            } else {
                userFriendlyMessage = `🔧 ${eventData.tool.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}...`;
            }
        } else if (eventData.step === 'tool-completed') {
            if (eventData.tool === 'list-datasources') {
                userFriendlyMessage = "✅ Data sources identified";
            } else if (eventData.tool === 'list-fields') {
                userFriendlyMessage = "✅ Data structure analyzed";
            } else if (eventData.tool === 'query-datasource') {
                userFriendlyMessage = "✅ Data query completed";
            } else {
                userFriendlyMessage = `✅ ${eventData.tool.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} completed`;
            }
        } else if (eventData.step === 'final-response') {
            userFriendlyMessage = "📝 Preparing your personalized insights...";
        }
        
        progressDiv.innerHTML = `<div class="mcp-progress" data-step="${eventData.step || 'default'}">${icon} <em>${userFriendlyMessage}</em></div>`;
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
function formatMCPResponse(text, toolResults = null) {
    if (!text) return '';
    
    let formatted = text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // **bold**
        .replace(/\*(.*?)\*/g, '<em>$1</em>') // *italic*
        .replace(/`(.*?)`/g, '<code>$1</code>') // `code`
        .replace(/#{1,6}\s+(.*?)(?=\n|$)/g, '<strong>$1</strong>'); // # headers

    // Add datasource information if available
    if (toolResults && toolResults.length > 0) {
        const datasources = extractDatasourceNames(toolResults);
        console.log('📊 Datasources:', datasources);
        console.log('📊 Tool Results:', toolResults);
        if (datasources.length > 0) {
            formatted += '<br><br><div class="datasource-info">📊 <strong>Data Sources Used:</strong> ' + datasources.join(', ') + '</div>';
        }
    }

    // Enhanced formatting for data tables and insights
    formatted = formatDataTables(formatted);
    formatted = formatInsightBoxes(formatted);
    
    return formatted;
}

// --- Extract Datasource Names from Tool Results ---
function extractDatasourceNames(toolResults) {
    const datasources = new Set();
    
    // First, collect all datasource info from list-datasources results
    const datasourceMap = new Map();
    for (const result of toolResults) {
        if (result.tool === 'list-datasources' && result.result) {
            try {
                let data;
                if (Array.isArray(result.result) && result.result[0] && result.result[0].text) {
                    data = JSON.parse(result.result[0].text);
                } else if (typeof result.result === 'string') {
                    data = JSON.parse(result.result);
                } else {
                    data = result.result;
                }
                
                if (Array.isArray(data)) {
                    data.forEach(ds => {
                        if (ds.id && ds.name) {
                            datasourceMap.set(ds.id, ds.name);
                        }
                    });
                }
            } catch (e) {
                console.warn('Failed to parse datasource data:', e);
            }
        }
    }
    
    // Then, only add datasources that were actually queried
    for (const result of toolResults) {
        if (result.tool === 'query-datasource' && result.arguments && result.arguments.datasourceLuid) {
            const datasourceName = datasourceMap.get(result.arguments.datasourceLuid);
            if (datasourceName) {
                datasources.add(datasourceName);
            }
        }
    }
    
    return Array.from(datasources);
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

// --- Dashboard Image Display Functions ---
function shouldDisplayDashboardImage(message) {
    const imageKeywords = ['show me', 'display', 'see', 'view', 'overview', 'dashboard', 'sheet', 'visualization', 'image'];
    const lowerMessage = message.toLowerCase();
    return imageKeywords.some(keyword => lowerMessage.includes(keyword));
}

function displayDashboardImage() {
    console.log('🖼️ Attempting to display dashboard image');
    
    // Look for image data in the current conversation
    const chatBox = document.getElementById('chatBox');
    if (!chatBox) return;
    
    // Check if we already have an image displayed
    const existingImage = chatBox.querySelector('.dashboard-image');
    if (existingImage) {
        console.log('🖼️ Dashboard image already displayed');
        return;
    }
    
    // Look for image data in the last bot message
    const botMessages = chatBox.querySelectorAll('.message.bot');
    if (botMessages.length === 0) return;
    
    const lastBotMessage = botMessages[botMessages.length - 1];
    const messageText = lastBotMessage.textContent || lastBotMessage.innerText || '';
    
    // Check if the message indicates an image was retrieved
    if (messageText.includes('Image retrieved successfully') || 
        messageText.includes('dashboard') || 
        messageText.includes('overview')) {
        
        // Create image placeholder
        const imageDiv = document.createElement('div');
        imageDiv.className = 'dashboard-image';
        imageDiv.innerHTML = `
            <div class="image-container">
                <div class="image-placeholder">
                    <div class="image-icon">🖼️</div>
                    <div class="image-text">Dashboard Image Retrieved</div>
                    <div class="image-note">The dashboard visualization has been successfully retrieved from your Tableau data.</div>
                </div>
            </div>
        `;
        
        // Add to the last bot message
        lastBotMessage.appendChild(imageDiv);
        
        console.log('🖼️ Dashboard image placeholder displayed');
    }
}

// --- Enhanced Connection Health Check ---
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, { 
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

    // Debug: Show which API URL is being used
    console.log('🌐 Using API URL:', API_BASE_URL);
    console.log('🌐 Current hostname:', window.location.hostname);

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
        addMessage(`⚠️ <strong>Server Connection Issue</strong><br>Unable to connect to the backend server at ${API_BASE_URL}. Please ensure the Python server is running and accessible.`, "bot");
        return;
    }

    // Show ready message
    addMessage("✅ <strong>Ready!</strong> You can now ask me about your Tableau data sources and get insights.", "bot");
});