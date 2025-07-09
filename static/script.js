// --- Simple chat functionality ---

let datasourceReady = false; // Track if LUID is ready

async function sendMessage() {
    if (!datasourceReady) {
        addMessage("Please wait for the datasource to finish loading.", "bot");
        return;
    }
    const input = document.getElementById('messageInput');
    if (!input) return;
    const message = input.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    input.value = '';

    const btn = document.getElementById('sendBtn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Thinking...';
    }

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        if (response.ok) {
            addMessage(data.response, 'bot');
        } else {
            // Show backend error message if available
            if (data && data.detail) {
                addMessage('Error: ' + data.detail, 'bot');
            } else {
                addMessage('Sorry, something went wrong! Please try again.', 'bot');
            }
        }

    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, I couldn\'t connect to the server. Please try again.', 'bot');
    }

    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Send';
    }
}

function addMessage(text, type) {
    const chatBox = document.getElementById('chatBox');
    if (!chatBox) return;
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.innerHTML = text.replace(/\n/g, '<br>');
    chatBox.appendChild(messageDiv);

    // Scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;
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
    // Disable input and button by default
    if (messageInput) messageInput.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    datasourceReady = false;

    try {
        await tableau.extensions.initializeAsync();

        const dashboard = tableau.extensions.dashboardContent.dashboard;
        const worksheets = dashboard.worksheets;
        const dataSourceMap = {}; // { name: LUID }

        // Collect all unique data sources by LUID
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
        } else {
            addMessage(
                "🔎 <b>Detected data sources in this dashboard:</b><br>" +
                namesArray.map(name => `• <b>${name}</b>`).join('<br>'),
                "bot"
            );
        }

        // Send the data source map to the backend for dynamic LUID selection
        const resp = await fetch('/datasources', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ datasources: dataSourceMap })
        });

        if (resp.ok) {
            datasourceReady = true;
            if (messageInput) messageInput.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
        } else {
            datasourceReady = false;
            if (messageInput) messageInput.disabled = true;
            if (sendBtn) sendBtn.disabled = true;
            addMessage("Could not initialize datasource. Please reload the dashboard.", "bot");
        }

    } catch (err) {
        datasourceReady = false;
        if (messageInput) messageInput.disabled = true;
        if (sendBtn) sendBtn.disabled = true;
        console.error("Error initializing Tableau Extensions API or fetching data sources:", err);
        addMessage("⚠️ Could not detect Tableau data sources. Make sure this extension is running inside a Tableau dashboard.", "bot");
    }
}

// --- Extension UI Resize Helpers ---
function resizeForChatOpen() {
    if (window.tableau && tableau.extensions && tableau.extensions.ui && tableau.extensions.ui.setSizeAsync) {
        tableau.extensions.ui.setSizeAsync({ width: 420, height: 600 });
    }
}

function resizeForChatClosed() {
    if (window.tableau && tableau.extensions && tableau.extensions.ui && tableau.extensions.ui.setSizeAsync) {
        tableau.extensions.ui.setSizeAsync({ width: 80, height: 80 });
    }
}

// --- Focus on input when page loads and set up chat UI ---
document.addEventListener('DOMContentLoaded', async function() {
    const chatIconBtn = document.getElementById('chatIconBtn');
    const chatContainer = document.getElementById('chatContainer');
    const closeChatBtn = document.getElementById('closeChatBtn');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    // Disable input/button at start
    if (messageInput) messageInput.disabled = true;
    if (sendBtn) sendBtn.disabled = true;

    // Keyboard enter handler
    if (messageInput) {
        messageInput.addEventListener('keypress', handleEnter);
    }

    // Send button handler
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }

    // Defensive: only add listeners if elements exist
    if (chatIconBtn && chatContainer && closeChatBtn) {
        // Try to initialize Tableau Extensions API
        try {
            await tableau.extensions.initializeAsync();
        } catch (e) {
            console.warn('Tableau Extensions API not available or not running in Tableau.');
        }
        // Start minimized (icon only)
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

    // --- Tableau extension: Detect, display, and send data sources on load ---
    listAndSendDashboardDataSources();
});
