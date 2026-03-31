// admin_script.js - Admin Portal for User & Group Management

const API_BASE_URL = window.API_BASE_URL || 'http://localhost:8001';
const CHAT_ENDPOINT = window.CHAT_ENDPOINT || '/admin-chat-stream';

let currentStream = null;
let conversationHistory = [];

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

    messageDiv.innerHTML = html;
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}

function showTypingIndicator(show = true) {
    const existingIndicator = document.getElementById('typing-indicator');
    if (show && !existingIndicator) {
        const indicatorHtml = `
            <div class="typing-indicator" id="typing-indicator">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
                <span class="typing-text">Processing...</span>
            </div>
        `;
        addMessage(indicatorHtml, 'bot', 'typing-indicator');
    } else if (!show && existingIndicator) {
        existingIndicator.remove();
    }
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) {
        input.focus();
        return;
    }

    if (currentStream) {
        addMessage("Please wait for the current operation to complete.", "bot");
        return;
    }

    // Display the user message but DON'T add to history yet
    // (will be added after we get the response)
    addMessage(escapeHtml(message), 'user');
    input.value = '';

    const sendBtn = document.getElementById('sendBtn');
    input.disabled = true;
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<span class="spinner"></span> Processing...';

    const botMessageId = 'bot-response-' + Date.now();

    try {
        const controller = new AbortController();
        currentStream = controller;

        await handleAdminRequest(message, controller, botMessageId);

    } catch (error) {
        console.error('Error:', error);
        showTypingIndicator(false);

        let errorMessage = "Error processing admin request.";
        if (error.name === 'AbortError') {
            errorMessage = "Request cancelled.";
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = "Connection error. Check server status.";
        }

        addMessage(errorMessage, 'bot', botMessageId);
    } finally {
        input.disabled = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = 'Send';
        currentStream = null;
    }
}

async function handleAdminRequest(message, controller, botMessageId) {
    const response = await fetch(`${API_BASE_URL}${CHAT_ENDPOINT}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        },
        body: JSON.stringify({ message, history: conversationHistory }),
        signal: controller.signal
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    await handleStreamingResponse(response, botMessageId, message);
}

async function handleStreamingResponse(response, botMessageId, userMessage) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    let progressDiv = null;

    const progressId = 'admin-progress-' + Date.now();
    progressDiv = addMessage('<div class="admin-progress"><em>Initializing...</em></div>', 'bot', progressId);

    let botMessageDiv = addMessage('', 'bot', botMessageId);
    if (botMessageDiv) {
        botMessageDiv.style.display = 'none';
    }

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split('\n\n');
            buffer = events.pop() || '';

            for (const eventBlock of events) {
                if (!eventBlock.trim()) continue;

                const lines = eventBlock.split('\n');
                let eventType = null;
                let eventData = null;

                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        eventType = line.substring(7).trim();
                    } else if (line.startsWith('data: ')) {
                        try {
                            eventData = JSON.parse(line.substring(6).trim());
                        } catch (e) {
                            continue;
                        }
                    }
                }

                if (eventType && eventData) {
                    if (eventType === 'progress' && progressDiv) {
                        progressDiv.innerHTML = `<div class="admin-progress"><em>${eventData.message || 'Processing...'}</em></div>`;
                    } else if (eventType === 'result') {
                        fullResponse = eventData.response || '';
                        if (fullResponse) {
                            botMessageDiv.innerHTML = formatAdminResponse(fullResponse, eventData.tool_results);
                            botMessageDiv.style.display = '';
                        }
                        showTypingIndicator(false);
                        if (progressDiv) {
                            progressDiv.remove();
                            progressDiv = null;
                        }
                        // NOW add both the user message and assistant response to history
                        // Include tool_calls if available for better context preservation
                        conversationHistory.push({ role: 'user', content: userMessage });
                        conversationHistory.push({
                            role: 'assistant',
                            content: fullResponse,
                            tool_calls: eventData.tool_calls || []
                        });
                    } else if (eventType === 'done') {
                        break;
                    }
                }
            }
        }

    } finally {
        showTypingIndicator(false);
        if (progressDiv) {
            progressDiv.remove();
        }
    }
}

function formatAdminResponse(text, toolResults = null) {
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

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function handleEnter(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function clearConversation() {
    if (confirm('Clear conversation history?')) {
        conversationHistory = [];
        const chatBox = document.getElementById('chatBox');
        if (chatBox) {
            // Keep only the welcome messages
            chatBox.innerHTML = `
                <div class="message bot">
                    👋 Welcome to the Tableau Admin Portal! I can help you manage users and groups on your Tableau Cloud site.
                    <br><br>
                    <strong>Try asking:</strong>
                    <br><br>
                    <strong>User Management:</strong>
                    <br>• "List all users on the site"
                    <br>• "Add a new user [user-email] with Creator role"
                    <br>• "Update user [user-email] to Viewer role"
                    <br>• "Show me all users with Explorer role"
                    <br><br>
                    <strong>Group Management:</strong>
                    <br>• "List all groups"
                    <br>• "Create a new group called 'Data Analysts'"
                    <br>• "Add user [user-email] to the Marketing group"
                    <br>• "Show me all users in the Sales group"
                    <br><br>
                    <em>Note: I specialize in user and group management only.</em>
                </div>
                <div class="message bot" style="background-color: #f0f9ff; border-left: 3px solid #0066cc;">
                    ✅ <strong>Ready!</strong> You can now ask me about user and group management operations.
                </div>
            `;
        }
        console.log('Conversation cleared');
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');

    console.log('Admin Portal - API URL:', API_BASE_URL);

    if (messageInput) messageInput.addEventListener('keypress', handleEnter);
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);
    if (clearBtn) clearBtn.addEventListener('click', clearConversation);

    if (messageInput) {
        setTimeout(() => messageInput.focus(), 100);
    }
});
