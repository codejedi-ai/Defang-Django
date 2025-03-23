$(document).ready(function() {
    // Send message when button is clicked
    $('#send-button').on('click', sendMessage);
    
    // Send message when Enter key is pressed
    $('#user-message').on('keypress', function(e) {
        if (e.which === 13) {
            sendMessage();
        }
    });
    
    function sendMessage() {
        const userMessage = $('#user-message').val().trim();
        
        if (userMessage === '') {
            return;
        }
        
        // Add user message to conversation
        $('#conversation-container').append(`
            <div class="user-bubble">
                ${userMessage}
            </div>
        `);
        
        // Clear input field
        $('#user-message').val('');
        
        // Show loading indicator
        $('#conversation-container').append(`
            <div class="ai-bubble" id="loading-bubble">
                <div class="loading"></div> Thinking...
            </div>
        `);
        
        // Scroll to bottom
        scrollToBottom();
        
        // Send request to AI bridge
        $.ajax({
            url: '/ai-bridge/',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                message: userMessage
            }),
            success: function(data) {
                // Remove loading indicator
                $('#loading-bubble').remove();
                
                // Add AI response to conversation
                $('#conversation-container').append(`
                    <div class="ai-bubble">
                        ${data.response}
                    </div>
                `);
                
                // Update history
                refreshHistory();
                
                // Scroll to bottom
                scrollToBottom();
            },
            error: function(xhr) {
                // Remove loading indicator
                $('#loading-bubble').remove();
                
                // Show error message
                $('#conversation-container').append(`
                    <div class="ai-bubble">
                        Sorry, there was an error processing your request.
                    </div>
                `);
                
                // Scroll to bottom
                scrollToBottom();
            }
        });
    }
    
    function scrollToBottom() {
        const container = document.getElementById('conversation-container');
        container.scrollTop = container.scrollHeight;
    }
    
    function refreshHistory() {
        $.ajax({
            url: '/conversation-history/',
            type: 'GET',
            success: function(data) {
                $('#history-container').empty();
                
                if (data.conversations.length === 0) {
                    $('#history-container').append('<p>No conversation history yet.</p>');
                    return;
                }
                
                data.conversations.forEach(function(conversation) {
                    $('#history-container').append(`
                        <div class="history-item" data-id="${conversation.id}">
                            <p class="history-user-message">${truncateText(conversation.user_message, 30)}</p>
                            <p class="history-ai-response">${truncateText(conversation.ai_response, 30)}</p>
                            <div class="timestamp">${conversation.created_at}</div>
                        </div>
                    `);
                });
            }
        });
    }
    
    function truncateText(text, maxLength) {
        if (text.length <= maxLength) {
            return text;
        }
        return text.substr(0, maxLength) + '...';
    }
});
