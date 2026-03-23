// script.js - With Tableau Extensions API for datasource detection when embedded in a dashboard

const API_BASE_URL = window.API_BASE_URL || (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://localhost:8000' 
    : 'https://mcpagent.up.railway.app');

/** Legacy workbook parameter name for viewer login (optional if using Dynamic User or user-context sheet). */
const TABLEAU_CURRENT_USER_PARAMETER = 'p_CurrentUser';
/** Your workbook may use this parameter name instead (spaces allowed in Tableau parameter names). */
const TABLEAU_DYNAMIC_USER_PARAMETER = 'Dynamic User';
/** Optional dedicated email parameter, e.g. USERNAME() on Cloud or USERATTRIBUTE('email') with Connected Apps. */
const TABLEAU_USER_EMAIL_PARAMETER = 'p_UserEmail';

/**
 * Dashboard worksheet name(s) for automatic per-viewer login (no parameter actions needed).
 * In Tableau: duplicate tab name to match; build sheet with calculated field = USERNAME() on Text (or Rows).
 * Add that sheet to this dashboard (can hide the tile). Tableau evaluates USERNAME() per signed-in user.
 */
/** Worksheet names for USERNAME()-on-Text pattern. "Dynamic User" is last to avoid clashing with the parameter name. */
const USER_CONTEXT_WORKSHEET_NAMES = [
    '_AnalyticsAgent_UserContext',
    'Analytics Agent User Context',
    'Dynamic User',
];

let datasourceReady = false;
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

function isLikelyEmail(str) {
    if (!str || typeof str !== 'string') return false;
    const s = str.trim();
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(s);
}

function setViewerIdentityDisplay(displayName, options = {}) {
    const el = document.getElementById('viewerIdentityValue');
    const labelEl = document.getElementById('viewerIdentityLabel');
    const subEl = document.getElementById('viewerIdentitySub');
    if (!el) return;
    const { placeholder = false, sub = null, label = null } = options;
    if (labelEl != null && label != null) {
        labelEl.textContent = label;
    }
    if (subEl) {
        if (sub) {
            subEl.textContent = sub;
            subEl.hidden = false;
        } else {
            subEl.textContent = '';
            subEl.hidden = true;
        }
    }
    const text =
        displayName != null && String(displayName).trim() !== '' ? String(displayName).trim() : '—';
    el.textContent = text;
    el.classList.toggle('viewer-identity-placeholder', placeholder || text === '—');
}

/** Gray header: only the workbook parameter "Dynamic User" (set from extension init). */
function setDynamicUserHeader(value) {
    const v = value != null && String(value).trim() !== '' ? String(value).trim() : null;
    setViewerIdentityDisplay(v, {
        label: 'Dynamic User',
        sub: null,
        placeholder: !v,
    });
}

function refreshViewerBarFromStoredIdentity() {
    if (window._tableauExtensionMode) {
        setDynamicUserHeader(window.extensionDynamicUserParameter);
        return;
    }
    const v = window.tableauLoggedInAs;
    if (v == null || String(v).trim() === '') {
        setViewerIdentityDisplay(null, { label: 'Signed in', sub: null, placeholder: true });
        return;
    }
    if (isLikelyEmail(v)) {
        setViewerIdentityDisplay(v.trim(), { label: 'Email', sub: null, placeholder: false });
        return;
    }
    setViewerIdentityDisplay(v, { label: 'Signed in', sub: null, placeholder: false });
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

    // When running as Tableau extension, require datasource init before sending
    if (window._tableauExtensionMode && !datasourceReady) {
        addMessage("⚠️ Please wait for the datasource to be initialized before asking questions.", "bot");
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
    sendBtn.innerHTML = '<span class="spinner"></span> Analyzing...';

    const botMessageId = 'bot-response-' + Date.now();
    let fullResponse = '';
    let hasStartedResponse = false;

    try {
        console.log('Creating AbortController');
        const controller = new AbortController();
        currentStream = controller;

        // Always use MCP/Analyst Agent endpoint
        console.log(`Using MCP/Analyst Agent for: "${message}"`);

        // Don't show thinking bubble immediately - wait for "planning" step
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
            body: JSON.stringify({ message }),
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

    // Keep thinking indicator visible - it will stay until we get actual results
    // Create progress indicator for MCP (shows detailed progress alongside thinking bubble)
    const progressId = 'mcp-progress-' + Date.now();
    progressDiv = addMessage('<div class="mcp-progress"><em>Initializing advanced analysis...</em></div>', 'bot', progressId);
    
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
                                if (eventData.data.logged_in_as != null || eventData.data.tableau_viewer_id != null) {
                                    window.tableauLoggedInAs = eventData.data.logged_in_as || eventData.data.tableau_viewer_id;
                                    refreshViewerBarFromStoredIdentity();
                                }
                            } else {
                                fullResponse = eventData.response || '';
                                // Store tool results for datasource info (new format)
                                if (eventData.tool_results) {
                                    window.lastToolResults = eventData.tool_results;
                                }
                                if (eventData.logged_in_as != null || eventData.tableau_viewer_id != null) {
                                    window.tableauLoggedInAs = eventData.logged_in_as || eventData.tableau_viewer_id;
                                    refreshViewerBarFromStoredIdentity();
                                }
                            }
                            console.log('Result event, fullResponse:', fullResponse);
                            console.log('Tool results for datasource extraction:', window.lastToolResults);
                            if (fullResponse) {
                                // Enhanced formatting for MCP results with datasource info
                                botMessageDiv.innerHTML = formatMCPResponse(fullResponse, window.lastToolResults);
                                // Show the message div now that we have content
                                botMessageDiv.style.display = '';
                            }
                            // Remove progress indicator and thinking indicator when we get results
                            showTypingIndicator(false);
                            if (progressDiv) {
                                progressDiv.remove();
                                progressDiv = null;
                            }
                            
                            // Auto-display dashboard image if user requested it
                            if (originalMessage && shouldDisplayDashboardImage(originalMessage)) {
                                console.log('Auto-displaying dashboard image based on user request');
                                setTimeout(() => displayDashboardImage(), 500);
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

// --- Handle MCP Progress Events ---
function handleMCPProgressEvent(eventData, progressDiv) {
    if (eventData.message) {
        let userFriendlyMessage = eventData.message;
        
        // Convert technical messages to user-friendly ones
        if (eventData.step === 'discover-tools') {
            userFriendlyMessage = "Discovering available data analysis tools...";
        } else if (eventData.step === 'tools-discovered') {
            userFriendlyMessage = `Found ${eventData.tools ? eventData.tools.length : 0} analysis tools ready to use`;
        } else if (eventData.step === 'iteration') {
            userFriendlyMessage = `Thinking through your question (Step ${eventData.iteration || 1})...`;
        } else if (eventData.step === 'executing-tools') {
            userFriendlyMessage = `Analyzing your data (${eventData.tool_count || 1} analysis step${eventData.tool_count > 1 ? 's' : ''})...`;
        } else if (eventData.step === 'tool-executing') {
            if (eventData.tool === 'list-datasources') {
                userFriendlyMessage = "Exploring available data sources...";
            } else if (eventData.tool === 'list-fields') {
                userFriendlyMessage = "Examining data structure and available fields...";
            } else if (eventData.tool === 'query-datasource') {
                userFriendlyMessage = "Querying data to find insights...";
            } else {
                userFriendlyMessage = `${eventData.tool.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}...`;
            }
        } else if (eventData.step === 'tool-completed') {
            if (eventData.tool === 'list-datasources') {
                userFriendlyMessage = "Data sources identified";
            } else if (eventData.tool === 'list-fields') {
                userFriendlyMessage = "Data structure analyzed";
            } else if (eventData.tool === 'query-datasource') {
                userFriendlyMessage = "Data query completed";
            } else {
                userFriendlyMessage = `${eventData.tool.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} completed`;
            }
        } else if (eventData.step === 'final-response') {
            userFriendlyMessage = "Preparing your personalized insights...";
        }
        
        progressDiv.innerHTML = `<div class="mcp-progress" data-step="${eventData.step || 'default'}"><em>${userFriendlyMessage}</em></div>`;
    }
}

// --- Get Progress Icon for MCP Steps --- (removed - no longer using icons)
function getMCPProgressIcon(step) {
    return '';
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
        console.log('Datasources:', datasources);
        console.log('Tool Results:', toolResults);
        if (datasources.length > 0) {
            formatted += '<br><br><div class="datasource-info"><strong>Data Sources Used:</strong> ' + datasources.join(', ') + '</div>';
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

/**
 * Tableau Extensions API: Parameter.currentValue is a DataValue ({ formattedValue, value, nativeValue, aliasValue }).
 * The Edit Parameter dialog often matches formattedValue; raw value may differ for special defaults (e.g. "User" on open).
 * Prefer a string that looks like an email when present; otherwise formatted → native → value → alias.
 */
function readParameterValueAsString(param) {
    if (!param) return null;
    const cv = param.currentValue != null ? param.currentValue : param.value;
    if (cv == null) return null;

    if (typeof cv === 'object') {
        const fv = typeof cv.formattedValue === 'string' ? cv.formattedValue.trim() : '';
        const nv = cv.nativeValue != null ? String(cv.nativeValue).trim() : '';
        const raw = cv.value != null ? String(cv.value).trim() : '';
        const av = cv.aliasValue != null ? String(cv.aliasValue).trim() : '';

        const parts = [fv, nv, raw, av].filter(Boolean);
        const asEmail = parts.find((s) => isLikelyEmail(s));
        if (asEmail) return asEmail;

        if (fv) return fv;
        if (nv) return nv;
        if (raw) return raw;
        if (av) return av;
        return null;
    }

    const s = String(cv).trim();
    if (!s || s === '[object Object]') return null;
    return s;
}

function normalizeParameterName(s) {
    if (s == null) return '';
    return String(s)
        .replace(/\u00a0/g, ' ')
        .trim()
        .toLowerCase()
        .replace(/\s+/g, ' ');
}

/**
 * Tableau's getParametersAsync returns parameters that are used in the workbook (on a viz, calc, filter, etc.).
 * findParameterAsync can miss when whitespace/casing differs; scanning the list is more reliable.
 */
async function findParameterFromDashboardList(dashboard, targetNormalized) {
    if (typeof dashboard.getParametersAsync !== 'function') return null;
    try {
        const all = await dashboard.getParametersAsync();
        if (!Array.isArray(all)) return null;
        for (const param of all) {
            if (!param || param.name == null) continue;
            if (normalizeParameterName(param.name) === targetNormalized) {
                return param;
            }
        }
    } catch (e) {
        console.warn('[Analytics Agent] getParametersAsync failed:', e);
    }
    return null;
}

async function resolveDashboardParameter(dashboard, exactName, targetNormalized) {
    if (exactName) {
        try {
            const p = await dashboard.findParameterAsync(exactName);
            if (p) return p;
        } catch (e) {
            console.warn('[Analytics Agent] findParameterAsync failed for', exactName, e);
        }
    }
    return findParameterFromDashboardList(dashboard, targetNormalized);
}

/** Log every parameter name the extension can see (helps when Dynamic User is missing). */
async function logVisibleParameterNames(dashboard) {
    if (typeof dashboard.getParametersAsync !== 'function') {
        console.warn('[Analytics Agent] getParametersAsync not available in this Tableau version.');
        return;
    }
    try {
        const all = await dashboard.getParametersAsync();
        const names = Array.isArray(all) ? all.map((p) => (p && p.name) || '?') : [];
        console.info(
            '[Analytics Agent] Parameters visible to the extension',
            names.length,
            '(unused parameters are often hidden—add a Parameter control on this dashboard):',
            names
        );
    } catch (e) {
        console.warn('[Analytics Agent] Could not list parameters:', e);
    }
}

/** Console-only: log DataValue fields so you can compare with Tableau's Edit Parameter dialog. */
function logParameterDataValueDebug(parameterName, param) {
    if (!param) return;
    const cv = param.currentValue != null ? param.currentValue : param.value;
    if (!cv || typeof cv !== 'object') {
        console.info('[Analytics Agent] Parameter', parameterName, 'raw:', cv);
        return;
    }
    console.info('[Analytics Agent] Parameter', parameterName, 'DataValue:', {
        formattedValue: cv.formattedValue,
        nativeValue: cv.nativeValue,
        value: cv.value,
        aliasValue: cv.aliasValue,
        resolved: readParameterValueAsString(param),
    });
}

function tableauSummaryCellToString(cell) {
    if (cell == null) return null;
    if (typeof cell === 'string' || typeof cell === 'number' || typeof cell === 'boolean') {
        const s = String(cell).trim();
        return s || null;
    }
    if (typeof cell === 'object') {
        if (cell.value != null && cell.value !== '') {
            const s = String(cell.value).trim();
            if (s) return s;
        }
        if (typeof cell.formattedValue === 'string' && cell.formattedValue.trim()) {
            return cell.formattedValue.trim();
        }
    }
    const s = String(cell).trim();
    if (s && s !== '[object Object]') return s;
    return null;
}

function extractLoginFromSummaryDataTable(dataTable) {
    if (!dataTable || !Array.isArray(dataTable.data) || dataTable.data.length === 0) {
        return null;
    }
    const columns = dataTable.columns || [];
    const prefer = /current\s*user|dynamic\s*user|username|user\s*name|login|email|p_/i;
    let colIdx = 0;
    for (let i = 0; i < columns.length; i++) {
        const col = columns[i];
        const cn = (col && (col.fieldName || col.name || col._fieldName)) || '';
        if (prefer.test(String(cn))) {
            colIdx = i;
            break;
        }
    }
    for (let r = 0; r < dataTable.data.length; r++) {
        const row = dataTable.data[r];
        if (!row || !row.length) continue;
        const cell = row[colIdx] != null ? row[colIdx] : row[0];
        const s = tableauSummaryCellToString(cell);
        if (s) return s;
    }
    return null;
}

/**
 * Reads per-viewer login from a dedicated worksheet (summary data = evaluated USERNAME() for current viewer).
 */
async function readLoginFromUserContextWorksheet(dashboard) {
    const wanted = new Set(USER_CONTEXT_WORKSHEET_NAMES.map((n) => n.trim().toLowerCase()));
    const worksheets = dashboard.worksheets || [];
    let target = null;
    for (const ws of worksheets) {
        if (ws && ws.name && wanted.has(String(ws.name).trim().toLowerCase())) {
            target = ws;
            break;
        }
    }
    if (!target) {
        console.info(
            '[Analytics Agent] No user-context worksheet. For dynamic USERNAME(), add a sheet named',
            USER_CONTEXT_WORKSHEET_NAMES[0],
            'with [currentUser] or USERNAME() on Text, and add it to this dashboard.'
        );
        return null;
    }

    let reader = null;
    try {
        if (typeof target.getSummaryDataReaderAsync === 'function') {
            reader = await target.getSummaryDataReaderAsync(1000, { ignoreSelection: true });
            const dataTable = await reader.getAllPagesAsync();
            const login = extractLoginFromSummaryDataTable(dataTable);
            if (login) {
                console.log('[Analytics Agent] Dynamic login from user-context sheet', target.name, ':', login);
            }
            return login;
        }
        const dataTable = await target.getSummaryDataAsync({
            maxRows: 100,
            ignoreSelection: true,
        });
        const login = extractLoginFromSummaryDataTable(dataTable);
        if (login) {
            console.log('[Analytics Agent] Dynamic login from user-context sheet', target.name, ':', login);
        }
        return login;
    } catch (e) {
        console.warn('[Analytics Agent] Could not read user-context worksheet', target.name, e);
        return null;
    } finally {
        if (reader && typeof reader.releaseAsync === 'function') {
            try {
                await reader.releaseAsync();
            } catch (releaseErr) {
                console.warn('[Analytics Agent] releaseAsync:', releaseErr);
            }
        }
    }
}

/**
 * Read workbook parameters for server/chat context (Dynamic User, p_CurrentUser, p_UserEmail).
 * `dynamicUserParameter` is only the "Dynamic User" parameter — used for the gray header.
 */
async function readWorkbookViewerParameters(dashboard) {
    let dynamicUserParameter = null;
    let currentUser = null;
    let userEmail = null;
    let currentUserFromParameter = null;
    const dynamicNorm = normalizeParameterName(TABLEAU_DYNAMIC_USER_PARAMETER);
    try {
        let dynamicUserParam = await resolveDashboardParameter(
            dashboard,
            TABLEAU_DYNAMIC_USER_PARAMETER,
            dynamicNorm
        );
        if (dynamicUserParam) {
            logParameterDataValueDebug(TABLEAU_DYNAMIC_USER_PARAMETER, dynamicUserParam);
        } else {
            console.warn(
                '[Analytics Agent] Could not find parameter',
                TABLEAU_DYNAMIC_USER_PARAMETER,
                '— checking visible parameters (see list below).'
            );
            await logVisibleParameterNames(dashboard);
        }
        dynamicUserParameter = readParameterValueAsString(dynamicUserParam);
        if (dynamicUserParam && !dynamicUserParameter) {
            console.warn(
                '[Analytics Agent] Parameter',
                TABLEAU_DYNAMIC_USER_PARAMETER,
                'exists but current value is empty in the API. Check DataValue log above and Tableau "Value when workbook opens".'
            );
        }
        currentUser = dynamicUserParameter;
        if (currentUser) {
            currentUserFromParameter = TABLEAU_DYNAMIC_USER_PARAMETER;
            console.log('[Analytics Agent] Parameter', TABLEAU_DYNAMIC_USER_PARAMETER, '=', currentUser);
        }
        if (!currentUser) {
            let userParam = await resolveDashboardParameter(
                dashboard,
                TABLEAU_CURRENT_USER_PARAMETER,
                normalizeParameterName(TABLEAU_CURRENT_USER_PARAMETER)
            );
            if (userParam) logParameterDataValueDebug(TABLEAU_CURRENT_USER_PARAMETER, userParam);
            currentUser = readParameterValueAsString(userParam);
            if (currentUser) {
                currentUserFromParameter = TABLEAU_CURRENT_USER_PARAMETER;
                console.log('[Analytics Agent] Parameter', TABLEAU_CURRENT_USER_PARAMETER, '=', currentUser);
            }
        }
        if (!currentUser) {
            console.info(
                '[Analytics Agent] No value from parameters',
                TABLEAU_DYNAMIC_USER_PARAMETER,
                'or',
                TABLEAU_CURRENT_USER_PARAMETER
            );
        }

        let emailParam = await resolveDashboardParameter(
            dashboard,
            TABLEAU_USER_EMAIL_PARAMETER,
            normalizeParameterName(TABLEAU_USER_EMAIL_PARAMETER)
        );
        if (emailParam) logParameterDataValueDebug(TABLEAU_USER_EMAIL_PARAMETER, emailParam);
        userEmail = readParameterValueAsString(emailParam);
        if (userEmail) {
            console.log('[Analytics Agent] Parameter', TABLEAU_USER_EMAIL_PARAMETER, '=', userEmail);
        }
    } catch (e) {
        console.warn('[Analytics Agent] readWorkbookViewerParameters failed:', e);
    }
    return { dynamicUserParameter, currentUser, userEmail, currentUserFromParameter };
}

/** Fire-and-forget POST to this app (same origin as extension) — audit, refresh hooks, etc. */
function silentTableauAction(user, action = 'audit_log') {
    const u = user == null || user === '' ? 'Unknown' : String(user);
    fetch(`${API_BASE_URL}/tableau-action`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Session-ID': generateSessionId(),
        },
        body: JSON.stringify({ user: u, action }),
    })
        .then(() => console.log('[Analytics Agent] tableau-action completed:', action))
        .catch((err) => console.warn('[Analytics Agent] tableau-action failed:', err));
}

// --- Tableau Extension: Data Source Detection ---
async function listAndSendDashboardDataSources() {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    if (messageInput) messageInput.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    datasourceReady = false;

    try {
        setViewerIdentityDisplay('Connecting…', { label: 'Dynamic User', sub: null, placeholder: true });
        addMessage("🔄 <em>Initializing Tableau Extensions API...</em>", "bot");

        await tableau.extensions.initializeAsync();

        const dashboard = tableau.extensions.dashboardContent.dashboard;
        const loginFromSheet = await readLoginFromUserContextWorksheet(dashboard);
        const {
            dynamicUserParameter,
            currentUser: workbookCurrentUser,
            userEmail: workbookUserEmail,
        } = await readWorkbookViewerParameters(dashboard);
        window.extensionDynamicUserParameter =
            dynamicUserParameter != null && String(dynamicUserParameter).trim() !== ''
                ? String(dynamicUserParameter).trim()
                : null;
        setDynamicUserHeader(window.extensionDynamicUserParameter);

        /** Sheet-evaluated USERNAME() wins over static parameters for server / chat context only. */
        const mergedWorkbookUser = loginFromSheet || workbookCurrentUser;
        const worksheets = dashboard.worksheets;
        const dataSourceMap = {};

        addMessage("📊 <em>Scanning dashboard for data sources...</em>", "bot");

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
            setDynamicUserHeader(window.extensionDynamicUserParameter);
            addMessage("⚠️ <strong>No data sources detected</strong><br>Please ensure this dashboard contains worksheets with connected data sources.", "bot");
            if (messageInput) messageInput.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            return;
        }

        // Collect all dashboard objects (worksheets, Pulse metrics, text, etc.) for context.
        // obj.id is the dashboard object/zone id (number); obj.type may be "pulse" or similar for Pulse cards.
        const dashboardObjects = (dashboard.objects || []).map(obj => ({
            id: obj.id != null ? obj.id : null,
            type: String(obj.type || ''),
            name: String(obj.name || '')
        }));
        console.log('Dashboard objects:', dashboardObjects);

        const dataSourceList = namesArray.map(name => `• <strong>${escapeHtml(name)}</strong>`).join('<br>');
        addMessage(`🔍 <strong>Found ${namesArray.length} data source(s):</strong><br>${dataSourceList}<br><br>⏳ <em>Initializing connection...</em>`, "bot");

        const resp = await fetch(`${API_BASE_URL}/datasources`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-Session-ID': generateSessionId()
            },
            body: JSON.stringify({
                datasources: dataSourceMap,
                dashboard_objects: dashboardObjects,
                workbook_current_user: mergedWorkbookUser,
                workbook_user_email: workbookUserEmail
            })
        });

        if (!resp.ok) {
            const errorData = await resp.json().catch(() => ({ detail: 'Unknown server error' }));
            throw new Error(errorData.detail || `Server error: ${resp.status} ${resp.statusText}`);
        }

        const respData = await resp.json();
        const datasourceLuid = respData.datasource_luid || dataSourceMap[namesArray[0]];
        const hasPulseObjects = !!respData.dashboard_has_pulse_objects;
        const loggedInAs = respData.logged_in_as || respData.tableau_viewer_id || null;
        window.tableauLoggedInAs = respData.viewer_email || loggedInAs;
        setDynamicUserHeader(window.extensionDynamicUserParameter);
        console.log('Datasource LUID (from extension):', datasourceLuid, 'dashboard_has_pulse_objects:', hasPulseObjects);
        if (loggedInAs) {
            console.log('Tableau viewer login (logged_in_as):', loggedInAs);
        }

        silentTableauAction(
            respData.viewer_email || loggedInAs || mergedWorkbookUser || 'Unknown',
            'audit_log'
        );

        datasourceReady = true;
        if (messageInput) {
            messageInput.disabled = false;
            messageInput.placeholder = "Ask me about your data insights...";
        }
        if (sendBtn) sendBtn.disabled = false;

        let readyHtml = `✅ <strong>Ready for intelligent analysis!</strong><br>Data source <strong>${escapeHtml(namesArray[0])}</strong> (LUID: <code>${escapeHtml(datasourceLuid)}</code>) is connected.`;
        if (!window.extensionDynamicUserParameter) {
            readyHtml +=
                '<br><br><small><strong>Dynamic User</strong> is empty: Tableau often only exposes parameters that are <strong>used</strong> on the workbook. Add a <strong>Parameter</strong> control for <code>Dynamic User</code> to this dashboard (or reference <code>[Parameters].[Dynamic User]</code> on a sheet), then reload. In the browser console, look for <code>[Analytics Agent] Parameters visible to the extension</code> to see exact names Tableau returned.</small>';
        }
        readyHtml += `<br><br>💡 <em>Try asking:</em><br>• "What are the top 3 insights from this data?"<br>• "What should I focus on to be proactive?"<br>• "List my Pulse metrics" (to see Pulse-based context)`;
        addMessage(readyHtml, "bot");

    } catch (err) {
        console.error("Initialization error:", err);
        if (window._tableauExtensionMode) {
            setDynamicUserHeader(window.extensionDynamicUserParameter);
        } else {
            setViewerIdentityDisplay(null, { placeholder: true });
        }
        let errorMessage = err.message;
        let suggestions = '';
        if (errorMessage.includes('Extensions API') || errorMessage.includes('tableau')) {
            suggestions = '<br><br>📋 <strong>Troubleshooting:</strong><br>• Ensure you\'re running this in a Tableau dashboard<br>• Check that the extension is properly configured';
        } else if (errorMessage.includes('server') || errorMessage.includes('fetch')) {
            suggestions = '<br><br>📋 <strong>Troubleshooting:</strong><br>• Check that the Python server is running<br>• Verify network connectivity';
        }
        addMessage(`❌ <strong>Initialization Failed</strong><br>${escapeHtml(errorMessage)}${suggestions}`, "bot");
        if (messageInput) messageInput.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
    }
}

// --- DOMContentLoaded Event Listener ---
document.addEventListener('DOMContentLoaded', async function() {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatContainer = document.getElementById('chatContainer');

    console.log('Using API URL:', API_BASE_URL);
    console.log('Current hostname:', window.location.hostname);

    if (messageInput) messageInput.addEventListener('keypress', handleEnter);
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);
    setupKeyboardShortcuts();

    const inTableauExtension = typeof tableau !== 'undefined' && tableau.extensions;
    window._tableauExtensionMode = !!inTableauExtension;
    if (!inTableauExtension) {
        datasourceReady = true;
        setViewerIdentityDisplay('Not in Tableau', { placeholder: true });
    }

    if (messageInput) setTimeout(() => messageInput.focus(), 100);

    const serverHealthy = await checkServerHealth();
    if (!serverHealthy) {
        setViewerIdentityDisplay('Server unreachable', { placeholder: true });
        addMessage(`<strong>Server Connection Issue</strong><br>Unable to connect to the backend at ${API_BASE_URL}. Please ensure the server is running.`, "bot");
        if (messageInput) messageInput.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        return;
    }

    if (inTableauExtension) {
        await listAndSendDashboardDataSources();
    } else {
        if (messageInput) messageInput.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        addMessage("<strong>Ready!</strong> You can now ask me about your Tableau data sources and get insights.", "bot");
    }
});