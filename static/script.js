// script.js - STANDALONE VERSION WITHOUT TABLEAU EXTENSIONS API

// Configuration - Set your deployed backend URL here
const API_BASE_URL = window.API_BASE_URL || (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : 'https://mcpagent.up.railway.app');

let currentStream = null;
let conversationHistory = []; // Track conversation for MCP context
let sessionId = null; // Track session ID for better client-server communication
let currentUser = null; // Track authenticated user

// --- Authentication Functions ---

// Check authentication status on page load
async function checkAuthStatus() {
    // Authentication disabled - go straight to chat interface
    console.log('Authentication bypassed - showing chat interface');
    showChatInterface();
}

// Show login interface
function showLoginInterface() {
    document.getElementById('loginContainer').style.display = 'flex';
    document.getElementById('chatContainer').style.display = 'none';
}

// Show chat interface
function showChatInterface() {
    document.getElementById('loginContainer').style.display = 'none';
    document.getElementById('chatContainer').style.display = 'block';
}

// Update user info display
function updateUserInfo(user) {
    const userInfoEl = document.getElementById('userInfo');
    if (userInfoEl && user) {
        userInfoEl.textContent = `${user.user_name} (${user.site_role})`;
    }
}

// Handle login
async function handleLogin() {
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;
    const loginBtn = document.getElementById('loginBtn');
    const errorDiv = document.getElementById('loginError');

    if (!username || !password) {
        errorDiv.textContent = 'Please enter both email and password';
        errorDiv.style.display = 'block';
        return;
    }

    // Update UI
    loginBtn.disabled = true;
    loginBtn.textContent = 'Signing in...';
    errorDiv.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE_URL}/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({
                username: username,
                password: password,
                use_pat: false
            })
        });

        const data = await response.json();

        if (data.success && data.is_admin) {
            currentUser = data;
            showChatInterface();
            updateUserInfo(data);
        } else {
            errorDiv.textContent = data.message || 'Authentication failed';
            errorDiv.style.display = 'block';
        }
    } catch (error) {
        errorDiv.textContent = `Connection error: ${error.message}`;
        errorDiv.style.display = 'block';
    } finally {
        loginBtn.disabled = false;
        loginBtn.textContent = 'Sign In';
    }
}

// Handle logout
async function handleLogout() {
    try {
        await fetch(`${API_BASE_URL}/logout`, {
            method: 'POST',
            credentials: 'include'
        });
    } catch (error) {
        console.error('Logout error:', error);
    }

    currentUser = null;
    conversationHistory = [];
    showLoginInterface();

    // Clear form
    document.getElementById('loginUsername').value = '';
    document.getElementById('loginPassword').value = '';
}

// Initialize auth on page load
document.addEventListener('DOMContentLoaded', () => {
    checkAuthStatus();

    // Login form handlers
    const loginBtn = document.getElementById('loginBtn');
    const loginPassword = document.getElementById('loginPassword');
    const logoutBtn = document.getElementById('logoutBtn');

    if (loginBtn) {
        loginBtn.addEventListener('click', handleLogin);
    }

    if (loginPassword) {
        loginPassword.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleLogin();
            }
        });
    }

    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
});

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

// --- Enhanced Progress Indicator / Thinking Bubble ---
function showTypingIndicator(show = true) {
    const existingIndicator = document.getElementById('typing-indicator');
    if (show && !existingIndicator) {
        const indicatorHtml = `
            <div class="typing-indicator" id="typing-indicator">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
                <span class="typing-text">Thinking...</span>
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

            console.log('Message:', message);

    // Prevent sending multiple messages simultaneously
    if (currentStream) {
        addMessage("Please wait for the current response to complete.", "bot");
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
    sendBtn.innerHTML = '<span class="spinner"></span> Processing...';

    const botMessageId = 'bot-response-' + Date.now();
    let fullResponse = '';
    let hasStartedResponse = false;

    try {
        console.log('Creating AbortController');
        const controller = new AbortController();
        currentStream = controller;

        // Always use Admin Agent endpoint
        console.log(`Using Admin Agent for: "${message}"`);

        console.log('Calling handleMCPRequest');
        fullResponse = await handleMCPRequest(message, controller, botMessageId);
        console.log('handleMCPRequest completed, response:', fullResponse);
        hasStartedResponse = true;

        // Display final response if we have one
        if (fullResponse && hasStartedResponse) {
            showTypingIndicator(false);
            // Response is already displayed by the streaming handler
            conversationHistory.push({ role: 'assistant', content: fullResponse });
        } else if (!hasStartedResponse) {
            // If nothing worked, show error
            showTypingIndicator(false);
            addMessage("I apologize, but I encountered an issue analyzing your request. Please try rephrasing your question or asking something more specific about your data.", 'bot', botMessageId);
        }

    } catch (error) {
        console.error('Error in sendMessage:', error);
        showTypingIndicator(false);
        
        // Enhanced error handling
        let errorMessage = "I encountered an error while processing your request.";
        
        if (error.name === 'AbortError') {
            errorMessage = "Request was cancelled.";
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = "Connection error. Please check your internet connection and try again.";
        } else if (error.message.includes('HTTP')) {
            errorMessage = `Server error: ${error.message}. Please try again later.`;
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
    console.log('handleMCPRequest called with message:', message);
    console.log('API_BASE_URL:', API_BASE_URL);
    
    try {
        const response = await fetch(`${API_BASE_URL}/mcp-chat-stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                'X-Session-ID': generateSessionId()
            },
            body: JSON.stringify({ message, history: conversationHistory }),
            signal: controller.signal
        });

        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);

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

        console.log('Calling handleMCPStreamingResponse');
        const result = await handleMCPStreamingResponse(response, botMessageId, message);
        console.log('handleMCPStreamingResponse completed, result:', result);
        return result;
    } catch (error) {
        console.error('Error in handleMCPRequest:', error);
        throw error;
    }
}

// --- Enhanced MCP Streaming Response Handler ---
async function handleMCPStreamingResponse(response, botMessageId, originalMessage = '') {
    console.log('handleMCPStreamingResponse called');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    let progressDiv = null;
    let isComplete = false;
    let timeoutId = null;

    // Create progress indicator for admin operations
    const progressId = 'admin-progress-' + Date.now();
    progressDiv = addMessage('<div class="admin-progress"><em>Initializing...</em></div>', 'bot', progressId);
    
    // Create bot message div but hide it until we have content
    let botMessageDiv = addMessage('', 'bot', botMessageId);
    if (botMessageDiv) {
        botMessageDiv.style.display = 'none';
    }

    try {
        console.log('Starting to read stream');
        
        // Keep thinking indicator visible until we get actual results
        // Don't remove it immediately - let progress events handle the transition
        
        // Set a timeout to prevent infinite analyzing state
        timeoutId = setTimeout(() => {
            if (!isComplete) {
                console.warn('Stream timeout - forcing completion');
                showTypingIndicator(false);
                if (progressDiv) {
                    progressDiv.remove();
                    progressDiv = null;
                }
                isComplete = true;
            }
        }, 120000); // 120 second timeout
        
        while (true && !isComplete) {
            const { done, value } = await reader.read();
            if (done) {
                console.log('Stream done');
                clearTimeout(timeoutId);
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            console.log('Buffer received:', buffer);
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

                    console.log('Event type:', eventType, 'Event data:', eventData);

                    if (eventType && eventData) {
                        if (eventType === 'progress' && progressDiv) {
                            handleMCPProgressEvent(eventData, progressDiv);
                        } else if (eventType === 'result') {
                            // Clear timeout when we get results
                            if (timeoutId) {
                                clearTimeout(timeoutId);
                                timeoutId = null;
                            }
                            
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
                            console.log('Result event, fullResponse:', fullResponse);
                            console.log('Tool results for datasource extraction:', window.lastToolResults);
                            if (fullResponse) {
                                botMessageDiv.innerHTML = formatMCPResponse(fullResponse);
                                // Show the message div now that we have content
                                botMessageDiv.style.display = '';
                            }
                            // Remove progress indicator and thinking indicator when we get results
                            showTypingIndicator(false);
                            if (progressDiv) {
                                progressDiv.remove();
                                progressDiv = null;
                            }
                            // Mark as complete but continue to wait for done event
                            isComplete = true;
                        } else if (eventType === 'error') {
                            throw new Error(eventData.error || 'Unknown MCP stream error');
                        } else if (eventType === 'done') {
                            console.log('Done event received');
                            // Clear timeout when done
                            if (timeoutId) {
                                clearTimeout(timeoutId);
                                timeoutId = null;
                            }
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

        console.log('Returning fullResponse:', fullResponse);
        return fullResponse;

    } finally {
        // Clear timeout if it exists
        if (timeoutId) {
            clearTimeout(timeoutId);
        }
        
        // Ensure progress indicator and thinking indicator are removed
        showTypingIndicator(false);
        if (progressDiv) {
            progressDiv.remove();
        }
    }
}

// --- Handle Admin Progress Events ---
function handleMCPProgressEvent(eventData, progressDiv) {
    if (eventData.message) {
        const raw = eventData.message || 'Processing...';
        const msg = escapeHtml(raw);
        const slackish = /slack/i.test(raw);
        const cls = slackish ? 'admin-progress admin-progress-slack' : 'admin-progress';
        const dot = slackish
            ? '<span class="progress-icon-slack" aria-hidden="true"></span>'
            : '<span class="progress-icon-tableau" aria-hidden="true"></span>';
        progressDiv.innerHTML = `<div class="${cls}">${dot}<em>${msg}</em></div>`;
    }
}

// --- Get Progress Icon for MCP Steps --- (removed - no longer using icons)
function getMCPProgressIcon(step) {
    return '';
}

// --- Admin Response Formatting ---
function formatMCPResponse(text) {
    if (!text) return '';

    let formatted = text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>');

    // Highlight admin operations
    formatted = formatted.replace(/✓/g, '<span style="color: #10b981;">✓</span>');
    formatted = formatted.replace(/✗/g, '<span style="color: #ef4444;">✗</span>');
    formatted = formatted.replace(/⚠️/g, '<span style="color: #f59e0b;">⚠️</span>');

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

// --- Enter / Shift+Enter (Slack-style multiline) ---
function handleEnter(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    } else if (event.key === 'Escape') {
        if (currentStream) {
            currentStream.abort();
            currentStream = null;
        }
    }
}

function autoResizeMessageInput() {
    const el = document.getElementById('messageInput');
    if (!el || el.tagName !== 'TEXTAREA') return;
    el.style.height = 'auto';
    const next = Math.min(el.scrollHeight, 200);
    el.style.height = `${Math.max(52, next)}px`;
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
            addMessage("<em>Request cancelled by user</em>", "bot");
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
    console.log('Attempting to display dashboard image');
    
    // Look for image data in the current conversation
    const chatBox = document.getElementById('chatBox');
    if (!chatBox) return;
    
    // Check if we already have an image displayed
    const existingImage = chatBox.querySelector('.dashboard-image');
    if (existingImage) {
        console.log('Dashboard image already displayed');
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
                    <div class="image-icon"></div>
                    <div class="image-text">Dashboard Image Retrieved</div>
                    <div class="image-note">The dashboard visualization has been successfully retrieved from your Tableau data.</div>
                </div>
            </div>
        `;
        
        // Add to the last bot message
        lastBotMessage.appendChild(imageDiv);
        
        console.log('Dashboard image placeholder displayed');
    }
}

// --- Health (includes Slack MCP flag for UI) ---
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
        if (!response.ok) {
            return { ok: false, slack: false };
        }
        const data = await response.json();
        return { ok: true, slack: !!data.slack_mcp_configured, raw: data };
    } catch (error) {
        console.warn('Server health check failed:', error);
        return { ok: false, slack: false };
    }
}

function updateIntegrationBar(health) {
    const el = document.getElementById('integrationBar');
    if (!el) return;
    if (!health.ok) {
        el.innerHTML =
            '<span class="badge badge-tableau">Tableau MCP</span><span class="integration-note integration-note-muted">Could not reach API health — check connection.</span>';
        return;
    }
    if (health.slack) {
        el.innerHTML =
            '<span class="badge badge-tableau">Tableau MCP</span><span class="badge badge-slack">Slack MCP</span><span class="integration-note">Bridge on: Tableau tools use native names; Slack tools use the <code>slack_mcp__</code> prefix.</span>';
    } else {
        el.innerHTML =
            '<span class="badge badge-tableau">Tableau MCP</span><span class="integration-note integration-note-muted">Slack bridge off (set <code>SLACK_MCP_SERVER</code> on the API to enable).</span>';
    }
}

// --- Enhanced DOMContentLoaded Event Listener ---
document.addEventListener('DOMContentLoaded', async function() {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    console.log('Using API URL:', API_BASE_URL);
    console.log('Current hostname:', window.location.hostname);

    if (messageInput) {
        messageInput.addEventListener('keydown', handleEnter);
        messageInput.addEventListener('input', autoResizeMessageInput);
    }
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);

    setupKeyboardShortcuts();

    if (messageInput) {
        setTimeout(() => messageInput.focus(), 100);
        autoResizeMessageInput();
    }

    const health = await checkServerHealth();
    updateIntegrationBar(health);
    if (!health.ok) {
        addMessage(
            `<strong>Server Connection Issue</strong><br>Unable to connect to the backend at ${escapeHtml(API_BASE_URL)}. Start the API server and refresh.`,
            'bot'
        );
        return;
    }

    const slackHint = health.slack
        ? ' Tableau and Slack MCP are both available from this UI.'
        : ' Slack MCP is not configured on the server; only Tableau admin tools are available here.';
    addMessage(
        `<div class="ready-banner"><strong>Ready.</strong>${escapeHtml(slackHint)} Ask in plain language—use <kbd>Enter</kbd> to send and <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line.</div>`,
        'bot'
    );
});