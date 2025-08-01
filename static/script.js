// script.js - REWRITTEN FOR STREAMING

let datasourceReady = false;

function addMessage(html, type, messageId = null) {
    const chatBox = document.getElementById('chatBox');
    if (!chatBox) return;

    let messageDiv;
    if (messageId && document.getElementById(messageId)) {
        // Update existing message div
        messageDiv = document.getElementById(messageId);
    } else {
        // Create new message div
        messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        if (messageId) {
            messageDiv.id = messageId;
        }
        chatBox.appendChild(messageDiv);
    }

    // Use innerHTML to render bold tags, etc.
    messageDiv.innerHTML = html.replace(/\n/g, '<br>');

    // Scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}

// --- NEW STREAMING SEND MESSAGE FUNCTION ---
async function sendMessage() {
    if (!datasourceReady) {
        addMessage("Please wait, the datasource is not ready.", "bot");
        return;
    }
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    input.value = '';
    
    // Disable input during response generation
    const sendBtn = document.getElementById('sendBtn');
    input.disabled = true;
    sendBtn.disabled = true;
    sendBtn.textContent = 'Thinking...';

    // Create a placeholder for the bot's response
    const botMessageId = 'bot-response-' + Date.now();
    let botMessageDiv = addMessage('<div class="loading-dots"><span>.</span><span>.</span><span>.</span></div>', 'bot', botMessageId);
    let fullResponse = '';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Network response was not ok');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep incomplete event in buffer

            for (const line of lines) {
                const eventMatch = line.match(/event: (.*)/);
                const dataMatch = line.match(/data: (.*)/);

                if (eventMatch && dataMatch) {
                    const event = eventMatch[1];
                    const data = JSON.parse(dataMatch[1]);

                    if (event === 'progress') {
                        // Display progress updates in the bot message bubble
                        botMessageDiv.innerHTML = `⏳ ${data.message}`;
                    } else if (event === 'token') {
                        // Append token to the response
                        fullResponse += data.token;
                        botMessageDiv.innerHTML = fullResponse.replace(/\n/g, '<br>');
                    } else if (event === 'result') {
                        // Final result received, ensure the full response is displayed
                        addMessage(data.response, 'bot', botMessageId); // Overwrite with final clean response
                    } else if (event === 'error') {
                        throw new Error(`Stream error: ${data.error}`);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage(`❌ **Error:**<br>${error.message}`, 'bot', botMessageId);
    } finally {
        // Re-enable input
        input.disabled = false;
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
        input.focus();
    }
}


function handleEnter(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// --- Your Existing Tableau Extensions and UI logic (no changes needed here) ---
async function listAndSendDashboardDataSources() {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    if (messageInput) messageInput.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    datasourceReady = false;

    try {
        await tableau.extensions.initializeAsync();
        const dashboard = tableau.extensions.dashboardContent.dashboard;
        const worksheets = dashboard.worksheets;
        const dataSourceMap = {};

        for (const worksheet of worksheets) {
            const dataSources = await worksheet.getDataSourcesAsync();
            dataSources.forEach(ds => {
                if (!Object.values(dataSourceMap).includes(ds.id)) {
                    dataSourceMap[ds.name] = ds.id;
                }
            });
        }
        
        const namesArray = Object.keys(dataSourceMap);
        if (namesArray.length === 0) {
            addMessage("No data sources detected in this dashboard.", "bot");
            return;
        }

        addMessage(`🔎 **Detected data source:**<br>• <b>${namesArray[0]}</b><br>Initializing...`, "bot");

        const resp = await fetch('/datasources', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ datasources: dataSourceMap })
        });
        
        const respData = await resp.json();
        if (resp.ok) {
            datasourceReady = true;
            if (messageInput) messageInput.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            addMessage(`✅ Ready! You can now ask questions about the <b>${namesArray[0]}</b> data source.`, "bot");
        } else {
            throw new Error(respData.detail || "Failed to initialize datasource on backend.");
        }
    } catch (err) {
        console.error("Error initializing Tableau:", err);
        addMessage(`⚠️ **Initialization Failed:**<br>${err.message}<br>Please ensure the extension is running in a Tableau dashboard and the Python server is running.`, "bot");
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('sendBtn').addEventListener('click', sendMessage);
    document.getElementById('messageInput').addEventListener('keypress', handleEnter);
    listAndSendDashboardDataSources();
});