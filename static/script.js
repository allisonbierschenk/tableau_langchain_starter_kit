// script.js - CORRECTED WITH UI AND STREAMING LOGIC

let datasourceReady = false;

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
    
    // Use innerHTML to render HTML tags like <br> and <b>
    messageDiv.innerHTML = html;
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}

// --- Streaming Chat Functionality ---
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
    
    const sendBtn = document.getElementById('sendBtn');
    input.disabled = true;
    sendBtn.disabled = true;
    sendBtn.textContent = 'Thinking...';

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
                        botMessageDiv.innerHTML = `⏳ ${data.message}`;
                    } else if (event === 'token') {
                        if (fullResponse.length === 0) botMessageDiv.innerHTML = ''; // Clear loading message
                        fullResponse += data.token;
                        botMessageDiv.innerHTML = fullResponse.replace(/\n/g, '<br>');
                    } else if (event === 'result') {
                        botMessageDiv.innerHTML = data.response.replace(/\n/g, '<br>');
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

// --- Tableau Extensions API: List, display, and send data sources ---
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

// --- RESTORED: Extension UI Resize Helpers ---
function resizeForChatOpen() {
    if (window.tableau?.extensions?.ui?.setSizeAsync) {
        tableau.extensions.ui.setSizeAsync({ width: 420, height: 600 });
    }
}

function resizeForChatClosed() {
    if (window.tableau?.extensions?.ui?.setSizeAsync) {
        tableau.extensions.ui.setSizeAsync({ width: 80, height: 80 });
    }
}

// --- RESTORED: Full DOMContentLoaded event listener ---
document.addEventListener('DOMContentLoaded', async function() {
    const chatIconBtn = document.getElementById('chatIconBtn');
    const chatContainer = document.getElementById('chatContainer');
    const closeChatBtn = document.getElementById('closeChatBtn');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    if (messageInput) messageInput.disabled = true;
    if (sendBtn) sendBtn.disabled = true;

    if (messageInput) messageInput.addEventListener('keypress', handleEnter);
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);

    if (chatIconBtn && chatContainer && closeChatBtn) {
        try {
            await tableau.extensions.initializeAsync();
        } catch (e) {
            console.warn('Tableau Extensions API not available.');
        }
        
        resizeForChatClosed();

        chatIconBtn.addEventListener('click', function() {
            chatContainer.classList.remove('chat-container-hidden');
            chatContainer.classList.add('chat-container-visible');
            chatIconBtn.style.display = 'none';
            resizeForChatOpen();
            setTimeout(() => { if (messageInput) messageInput.focus(); }, 100);
        });

        closeChatBtn.addEventListener('click', function() {
            chatContainer.classList.remove('chat-container-visible');
            chatContainer.classList.add('chat-container-hidden');
            chatIconBtn.style.display = 'flex';
            resizeForChatClosed();
        });
    }

    if (messageInput && !chatContainer.classList.contains('chat-container-hidden')) {
        messageInput.focus();
    }

    listAndSendDashboardDataSources();
});