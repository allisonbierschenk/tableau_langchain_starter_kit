// script.js - ENHANCED VERSION WITH BETTER ERROR HANDLING AND UX

let datasourceReady = false;
let currentStream = null;

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
                <span class="typing-text">AI is thinking...</span>
            </div>
        `;
        addMessage(indicatorHtml, 'bot', 'typing-indicator');
    } else if (!show && existingIndicator) {
        existingIndicator.remove();
    }
}

// --- Enhanced Streaming Chat Functionality ---
async function sendMessage() {
    // Validation checks
    if (!datasourceReady) {
        addMessage("⚠️ Please wait for the datasource to be initialized before asking questions.", "bot");
        return;
    }

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

    // Display user message
    addMessage(escapeHtml(message), 'user');
    input.value = '';
    
    // Update UI state
    const sendBtn = document.getElementById('sendBtn');
    input.disabled = true;
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<span class="spinner"></span> Thinking...';

    const botMessageId = 'bot-response-' + Date.now();
    let botMessageDiv = showTypingIndicator();
    let fullResponse = '';
    let hasStartedResponse = false;

    try {
        const controller = new AbortController();
        currentStream = controller;

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
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

        if (!response.body) {
            throw new Error('No response body received from server');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // Remove typing indicator when we start receiving real data
        showTypingIndicator(false);
        botMessageDiv = addMessage('', 'bot', botMessageId);

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
                                eventData = JSON.parse(dataStr);
                            }
                        }
                    }

                    if (eventType && eventData) {
                        await handleStreamEvent(eventType, eventData, botMessageDiv, fullResponse);
                        
                        if (eventType === 'token') {
                            if (!hasStartedResponse) {
                                hasStartedResponse = true;
                                botMessageDiv.innerHTML = ''; // Clear any loading content
                            }
                            fullResponse += eventData.token;
                            // Enhanced formatting for better readability
                            botMessageDiv.innerHTML = formatResponse(fullResponse);
                        } else if (eventType === 'result') {
                            botMessageDiv.innerHTML = formatResponse(eventData.response);
                            break;
                        } else if (eventType === 'done') {
                            break;
                        }
                    }
                } catch (parseError) {
                    console.warn('Failed to parse event:', eventBlock, parseError);
                    continue;
                }
            }
        }

        // Ensure we have some response
        if (!hasStartedResponse && !botMessageDiv.innerHTML.trim()) {
            botMessageDiv.innerHTML = "I completed processing your request, but didn't generate a visible response. Please try rephrasing your question.";
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
        
        addMessage(`❌ <strong>Error:</strong><br>${escapeHtml(errorMessage)}`, 'bot', botMessageId);
    } finally {
        // Reset UI state
        currentStream = null;
        input.disabled = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = 'Send';
        input.focus();
    }
}

// --- Enhanced Stream Event Handling ---
async function handleStreamEvent(eventType, eventData, messageDiv, currentResponse) {
    switch (eventType) {
        case 'progress':
            if (eventData.message) {
                messageDiv.innerHTML = `⏳ <em>${escapeHtml(eventData.message)}</em>`;
            }
            break;
        case 'error':
            throw new Error(eventData.error || 'Unknown stream error');
        case 'token':
            // Handled in main loop
            break;
        case 'result':
            // Final result - handled in main loop
            break;
        case 'done':
            // Stream complete
            break;
        default:
            console.debug('Unknown event type:', eventType, eventData);
    }
}

// --- Response Formatting Helper ---
function formatResponse(text) {
    if (!text) return '';
    
    return text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // **bold**
        .replace(/\*(.*?)\*/g, '<em>$1</em>') // *italic*
        .replace(/`(.*?)`/g, '<code>$1</code>') // `code`
        .replace(/#{1,6}\s+(.*?)(?=\n|$)/g, '<strong>$1</strong>'); // # headers
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

// --- Enhanced Tableau Data Source Management ---
async function listAndSendDashboardDataSources() {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    
    // Disable inputs during initialization
    if (messageInput) messageInput.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    datasourceReady = false;

    try {
        addMessage("🔄 <em>Initializing Tableau Extensions API...</em>", "bot");
        
        await tableau.extensions.initializeAsync();
        const dashboard = tableau.extensions.dashboardContent.dashboard;
        const worksheets = dashboard.worksheets;
        const dataSourceMap = {};

        addMessage("📊 <em>Scanning dashboard for data sources...</em>", "bot");

        // Collect all unique data sources
        for (const worksheet of worksheets) {
            try {
                const dataSources = await worksheet.getDataSourcesAsync();
                dataSources.forEach(ds => {
                    if (!Object.values(dataSourceMap).includes(ds.id)) {
                        dataSourceMap[ds.name] = ds.id;
                    }
                });
            } catch (wsError) {
                console.warn(`Failed to get data sources from worksheet ${worksheet.name}:`, wsError);
            }
        }
        
        const namesArray = Object.keys(dataSourceMap);
        if (namesArray.length === 0) {
            addMessage("⚠️ <strong>No data sources detected</strong><br>Please ensure this dashboard contains worksheets with connected data sources.", "bot");
            return;
        }

        // Display found data sources
        const dataSourceList = namesArray.map(name => `• <strong>${escapeHtml(name)}</strong>`).join('<br>');
        addMessage(`🔍 <strong>Found ${namesArray.length} data source(s):</strong><br>${dataSourceList}<br><br>⏳ <em>Initializing connection...</em>`, "bot");

        // Send to backend
        const resp = await fetch('/datasources', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ datasources: dataSourceMap })
        });
        
        if (!resp.ok) {
            const errorData = await resp.json().catch(() => ({ detail: 'Unknown server error' }));
            throw new Error(errorData.detail || `Server error: ${resp.status} ${resp.statusText}`);
        }

        const respData = await resp.json();
        datasourceReady = true;
        
        // Re-enable inputs
        if (messageInput) {
            messageInput.disabled = false;
            messageInput.placeholder = "Ask questions about your data...";
        }
        if (sendBtn) sendBtn.disabled = false;

        addMessage(`✅ <strong>Ready!</strong><br>You can now ask questions about your data. The primary data source <strong>${escapeHtml(namesArray[0])}</strong> is active.<br><br>💡 <em>Try asking: "What are the total number of records?" or "Show me a summary of the data"</em>`, "bot");

    } catch (err) {
        console.error("Initialization error:", err);
        
        let errorMessage = err.message;
        let suggestions = '';
        
        if (errorMessage.includes('Extensions API')) {
            suggestions = '<br><br>📋 <strong>Troubleshooting:</strong><br>• Ensure you\'re running this in a Tableau dashboard<br>• Check that the extension is properly configured<br>• Verify the manifest file is accessible';
        } else if (errorMessage.includes('server') || errorMessage.includes('fetch')) {
            suggestions = '<br><br>📋 <strong>Troubleshooting:</strong><br>• Check that the Python server is running<br>• Verify network connectivity<br>• Check server logs for errors';
        }
        
        addMessage(`❌ <strong>Initialization Failed</strong><br>${escapeHtml(errorMessage)}${suggestions}`, "bot");
    }
}

// --- Enhanced UI Resize Helpers ---
function resizeForChatOpen() {
    if (window.tableau?.extensions?.ui?.setSizeAsync) {
        tableau.extensions.ui.setSizeAsync({ width: 450, height: 650 }).catch(console.warn);
    }
}

function resizeForChatClosed() {
    if (window.tableau?.extensions?.ui?.setSizeAsync) {
        tableau.extensions.ui.setSizeAsync({ width: 80, height: 80 }).catch(console.warn);
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
    const chatIconBtn = document.getElementById('chatIconBtn');
    const chatContainer = document.getElementById('chatContainer');
    const closeChatBtn = document.getElementById('closeChatBtn');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    // Initial UI state
    if (messageInput) {
        messageInput.disabled = true;
        messageInput.placeholder = "Initializing...";
    }
    if (sendBtn) sendBtn.disabled = true;

    // Setup event listeners
    if (messageInput) messageInput.addEventListener('keypress', handleEnter);
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);
    
    setupKeyboardShortcuts();

    // Chat UI management
    if (chatIconBtn && chatContainer && closeChatBtn) {
        // Initialize Tableau Extensions API
        try {
            await tableau.extensions.initializeAsync();
        } catch (e) {
            console.warn('Tableau Extensions API not available:', e);
        }
        
        resizeForChatClosed();

        chatIconBtn.addEventListener('click', function() {
            chatContainer.classList.remove('chat-container-hidden');
            chatContainer.classList.add('chat-container-visible');
            chatIconBtn.style.display = 'none';
            resizeForChatOpen();
            setTimeout(() => { 
                if (messageInput && !messageInput.disabled) messageInput.focus(); 
            }, 100);
        });

        closeChatBtn.addEventListener('click', function() {
            chatContainer.classList.remove('chat-container-visible');
            chatContainer.classList.add('chat-container-hidden');
            chatIconBtn.style.display = 'flex';
            resizeForChatClosed();
            
            // Cancel any ongoing streams
            if (currentStream) {
                currentStream.abort();
                currentStream = null;
            }
        });
    }

    // Auto-focus if chat is already open
    if (messageInput && !chatContainer?.classList.contains('chat-container-hidden')) {
        setTimeout(() => messageInput.focus(), 100);
    }

    // Check server health before initializing
    const serverHealthy = await checkServerHealth();
    if (!serverHealthy) {
        addMessage("⚠️ <strong>Server Connection Issue</strong><br>Unable to connect to the backend server. Please ensure the Python server is running and accessible.", "bot");
        return;
    }

    // Initialize data sources
    await listAndSendDashboardDataSources();
});