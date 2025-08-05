// script.js (REVISED)

let datasourceReady = false;
let currentStreamController = null;
let conversationHistory = []; // Keep track of the full conversation

// --- Message Display Helper ---
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
    
    // Use a library like DOMPurify in a real app to prevent XSS
    messageDiv.innerHTML = formatResponse(html);
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}

// --- Typing Indicator ---
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

// --- Main Send Message Function ---
async function sendMessage() {
    if (!datasourceReady) {
        addMessage("⚠️ Please wait for the datasource to be initialized.", "bot");
        return;
    }

    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) return;

    if (currentStreamController) {
        addMessage("⏳ Please wait for the current response to complete.", "bot");
        return;
    }

    // Add user message to UI and history
    addMessage(message, 'user');
    conversationHistory.push({ role: 'user', content: message });
    input.value = '';
    
    // Update UI state
    const sendBtn = document.getElementById('sendBtn');
    input.disabled = true;
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<span class="spinner"></span> Thinking...';

    const botMessageId = 'bot-response-' + Date.now();
    let botMessageDiv = null;
    showTypingIndicator(true);

    try {
        currentStreamController = new AbortController();

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            },
            body: JSON.stringify({
                message: message,
                history: conversationHistory.slice(0, -1) // Send history without the current message
            }),
            signal: currentStreamController.signal
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Server error' }));
            throw new Error(errorData.detail);
        }

        await handleStreamingResponse(response, botMessageId);

    } catch (error) {
        console.error('Chat error:', error);
        let errorMessage = (error.name === 'AbortError') ? 'Request was cancelled.' : error.message;
        addMessage(`❌ <strong>Error:</strong><br>${escapeHtml(errorMessage)}`, 'bot', botMessageId);
    } finally {
        // Reset UI state
        showTypingIndicator(false);
        currentStreamController = null;
        input.disabled = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = 'Send';
        input.focus();
    }
}

// --- Handle Streaming Response ---
async function handleStreamingResponse(response, botMessageId) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalResponse = '';
    let botMessageDiv = null;

    showTypingIndicator(false); // Hide "thinking" indicator, we'll stream the answer
    botMessageDiv = addMessage("✍️", 'bot', botMessageId); // Placeholder for the streaming response

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || ''; // Keep incomplete event in buffer

        for (const eventBlock of events) {
            if (!eventBlock.trim()) continue;

            const eventType = eventBlock.match(/event: (.*)/)?.[1];
            const eventDataStr = eventBlock.match(/data: (.*)/)?.[1];

            if (eventType === 'result' && eventDataStr) {
                const data = JSON.parse(eventDataStr);
                finalResponse = data.response;
                botMessageDiv.innerHTML = formatResponse(finalResponse);
            } else if (eventType === 'error' && eventDataStr) {
                const data = JSON.parse(eventDataStr);
                throw new Error(data.error);
            }
        }
    }

    if (finalResponse) {
        conversationHistory.push({ role: 'assistant', content: finalResponse });
    }
}


// --- Formatting and Utility Helpers ---
function formatResponse(text) {
    if (!text) return '';
    
    let formatted = escapeHtml(text)
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>');

    return formatDataTables(formatted);
}

function formatDataTables(text) {
    const tableRegex = /(\|.+\|\s*<br>\|[-:\s|]+\|\s*<br>(?:\|.+\|<br>?)*)/g;
    
    return text.replace(tableRegex, (match) => {
        const lines = match.trim().split('<br>').filter(line => line.trim());
        if (lines.length < 3) return match;
        
        let tableHtml = '<div class="data-table-wrapper"><table class="data-table">';
        // Header
        const headerCells = lines[0].split('|').slice(1, -1).map(cell => cell.trim());
        tableHtml += '<thead><tr>' + headerCells.map(h => `<th>${h}</th>`).join('') + '</tr></thead>';
        
        // Body
        tableHtml += '<tbody>';
        lines.slice(2).forEach(line => {
            const rowCells = line.split('|').slice(1, -1).map(cell => cell.trim());
            tableHtml += '<tr>' + rowCells.map(c => `<td>${c}</td>`).join('') + '</tr>';
        });
        tableHtml += '</tbody></table></div>';
        
        return tableHtml;
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function handleEnter(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    } else if (event.key === 'Escape' && currentStreamController) {
        currentStreamController.abort();
    }
}

// --- Tableau Initialization (remains mostly the same) ---
async function listAndSendDashboardDataSources() {
    // This function's logic is sound and does not need to be changed.
    // It correctly initializes the extension, finds datasources, and sends them to the backend.
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    
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

        for (const worksheet of worksheets) {
            const dataSources = await worksheet.getDataSourcesAsync();
            dataSources.forEach(ds => {
                if (!dataSourceMap[ds.name]) {
                    dataSourceMap[ds.name] = ds.id;
                }
            });
        }
        
        const namesArray = Object.keys(dataSourceMap);
        if (namesArray.length === 0) {
            addMessage("⚠️ <strong>No data sources detected.</strong>", "bot");
            return;
        }

        const dataSourceList = namesArray.map(name => `• <strong>${escapeHtml(name)}</strong>`).join('<br>');
        addMessage(`🔍 <strong>Found ${namesArray.length} data source(s):</strong><br>${dataSourceList}<br><br>⏳ <em>Initializing connection...</em>`, "bot");

        const resp = await fetch('/datasources', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ datasources: dataSourceMap })
        });
        
        if (!resp.ok) {
            const errorData = await resp.json().catch(() => ({ detail: 'Unknown server error' }));
            throw new Error(errorData.detail);
        }

        await resp.json();
        datasourceReady = true;
        
        if (messageInput) {
            messageInput.disabled = false;
            messageInput.placeholder = "Ask questions about your data...";
        }
        if (sendBtn) sendBtn.disabled = false;

        addMessage(`✅ <strong>Ready!</strong><br>You can now ask questions about the <strong>${escapeHtml(namesArray[0])}</strong> data source.`, "bot");

    } catch (err) {
        console.error("Initialization error:", err);
        addMessage(`❌ <strong>Initialization Failed</strong><br>${escapeHtml(err.message)}`, "bot");
    }
}

// --- DOMContentLoaded (remains the same) ---
document.addEventListener('DOMContentLoaded', async function() {
    // This setup logic is sound and does not need to be changed.
    // It correctly handles UI elements and triggers initialization.
    document.getElementById('messageInput').addEventListener('keypress', handleEnter);
    document.getElementById('sendBtn').addEventListener('click', sendMessage);
    
    // Check for Tableau environment before initializing
    if (window.tableau?.extensions) {
        await listAndSendDashboardDataSources();
    } else {
        addMessage("⚠️ This app is designed to run as a Tableau Extension. Some features may not work.", "bot");
        // For local testing without Tableau, you can manually enable the UI
        document.getElementById('messageInput').disabled = false;
        document.getElementById('sendBtn').disabled = false;
        datasourceReady = true; // Set to true for local testing, but datasources won't work.
    }
});