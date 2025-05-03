import React, { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
// Removed: import { supabase } from "../utils/supabaseClient"; // No longer needed
import ChatFeed from "./ChatFeed";
import InputArea from "./InputArea";
import { Copy, RefreshCw } from "lucide-react";
import "../styles/ChatInterface.css";
import "../styles/utils/_overlay.css";

// Define WebSocket URL (Make sure this matches your server.js port)
const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || "ws://localhost:3001";

// Component to handle individual chat sessions
function ChatPage({ systemInfo, user, refreshHistory, selectedModel }) {
  const { chatId } = useParams();
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true); // Separate loading state for history
  const [showPlaceholder, setShowPlaceholder] = useState(false); // Initial state depends on loading
  const [error, setError] = useState(null);
  const [streamingAssistantMessage, setStreamingAssistantMessage] = useState(null);
  const [copySuccess, setCopySuccess] = useState(""); // State for copy feedback
  const bottomRef = useRef(null);
  const ws = useRef(null);
  const currentAssistantMessageId = useRef(null);
  const accumulatedContentRef = useRef("");
  const isConnected = useRef(false); // Track WebSocket connection status

  // --- WebSocket Connection and Message Request ---
  useEffect(() => {
    setIsLoadingHistory(true); // Start loading history
    setError(null); // Clear previous errors
    setMessages([]); // Clear messages from previous chat
    setShowPlaceholder(false); // Hide placeholder while loading

    // Function to connect WebSocket
    const connectWebSocket = () => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
         console.log("WebSocket already connected.");
         // If already connected, request messages for the new chat ID
         requestMessagesForChat(chatId);
         return;
      }

      console.log(`Attempting to connect WebSocket to ${WEBSOCKET_URL}...`);
      ws.current = new WebSocket(WEBSOCKET_URL);

      ws.current.onopen = () => {
        console.log("WebSocket Connected");
        isConnected.current = true;
        setError(null); // Clear connection errors
        // Request messages for the current chat ID upon successful connection
        requestMessagesForChat(chatId);
      };

      ws.current.onmessage = (event) => {
        handleWebSocketMessage(event.data);
      };

      ws.current.onerror = (err) => {
        console.error("WebSocket Error:", err);
        setError("WebSocket connection error. Please try refreshing.");
        isConnected.current = false;
        setIsLoadingHistory(false); // Stop loading on error
        setShowPlaceholder(true); // Show placeholder on connection error
      };

      ws.current.onclose = () => {
        console.log("WebSocket Disconnected");
        isConnected.current = false;
        // Optionally notify user or attempt reconnect
        // If it disconnects while loading history, handle appropriately
        if (isLoadingHistory) {
           setError("WebSocket disconnected while loading history.");
           setIsLoadingHistory(false);
           setShowPlaceholder(true);
        }
      };
    };

    // Function to request messages via WebSocket
    const requestMessagesForChat = (currentChatId) => {
       if (ws.current && ws.current.readyState === WebSocket.OPEN && currentChatId) {
           console.log(`Requesting messages for chat ID: ${currentChatId}`);
           ws.current.send(JSON.stringify({
               type: "get_messages",
               payload: { chatId: currentChatId }
           }));
           setIsLoadingHistory(true); // Ensure loading state is active
       } else if (!currentChatId) {
            console.warn("Cannot request messages: No chat ID.");
            setIsLoadingHistory(false);
            setShowPlaceholder(true);
       } else {
           console.warn("Cannot request messages: WebSocket not connected.");
           // Attempt to reconnect or show error
           setError("WebSocket not connected. Cannot load messages.");
           setIsLoadingHistory(false);
           setShowPlaceholder(true);
       }
    };

    // Initialize connection
    connectWebSocket();

    // Cleanup WebSocket on component unmount
    return () => {
      console.log("Closing WebSocket connection.");
      ws.current?.close();
      isConnected.current = false;
    };
  }, [chatId]); // Re-run effect when chatId changes


  // Scroll to bottom when messages change or while streaming
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streamingAssistantMessage]);


  // --- WebSocket Message Handler ---
  const handleWebSocketMessage = useCallback((data) => {
    try {
      const message = JSON.parse(data);
      // console.log("WS Message Received:", message.type, message.payload); // Debug log

      switch (message.type) {
        // --- Handle receiving initial messages for the chat ---
        case "chat_history":
            if (message.payload.chatId === chatId) {
                console.log(`Received history for chat ${chatId}:`, message.payload.messages);
                setMessages(message.payload.messages || []);
                setShowPlaceholder(!message.payload.messages || message.payload.messages.length === 0);
            } else {
                console.warn(`Received history for wrong chat ID (${message.payload.chatId}), expected ${chatId}.`);
            }
            setIsLoadingHistory(false); // History loading finished
            break;
        case "chat_history_error":
            console.error("Error loading chat history from backend:", message.payload.error);
            setError(`Failed to load chat history: ${message.payload.error}`);
            setMessages([]);
            setShowPlaceholder(true);
            setIsLoadingHistory(false);
            break;

        // --- Handle streaming assistant response ---
        case "chunk":
          setIsGenerating(true);
          const contentChunk = message.payload.content;
          accumulatedContentRef.current += contentChunk;

          setStreamingAssistantMessage((prev) => {
            if (!prev && currentAssistantMessageId.current) {
              return {
                id: currentAssistantMessageId.current, // Use the temporary ID
                sender: "assistant",
                content: accumulatedContentRef.current,
                chat_id: chatId,
                created_at: new Date().toISOString(), // Initial timestamp
                isLoading: true,
              };
            } else if (prev) {
              return { ...prev, content: accumulatedContentRef.current };
            }
            return prev; // Should have an ID if receiving chunks
          });
          break;
        case "end":
          setIsGenerating(false);
          const finalContent = accumulatedContentRef.current;

          if (finalContent && currentAssistantMessageId.current) {
            // Final message object to add to state
            const finalMessage = {
              id: currentAssistantMessageId.current, // Use the temporary ID
              sender: "assistant",
              content: finalContent,
              chat_id: chatId,
              created_at: streamingAssistantMessage?.created_at || new Date().toISOString(), // Use start time if available
              isLoading: false,
            };

            // Add the final message to the main messages list
            // Replace the temporary message if it exists, otherwise append
            setMessages((prev) => {
                const existing = prev.find(msg => msg.id === finalMessage.id);
                if (existing) {
                    return prev.map(msg => msg.id === finalMessage.id ? finalMessage : msg);
                } else {
                    // If streaming message wasn't in main list yet, add it
                    return [...prev, finalMessage];
                }
            });

            // NOTE: Backend is now responsible for saving the assistant message.
            // We still refresh history here if it's the first assistant response
            // to potentially pick up a title generated by the backend.
            const isFirstAssistantResponse = messages.filter(m => !m.isLoading).length === 1 && messages[0].sender === 'user';
            if (isFirstAssistantResponse) {
                 console.log("First assistant response complete, refreshing history for potential title.");
                 refreshHistory();
            }

          } else {
            console.log("End event received but no content accumulated or ID tracked.");
          }

          // Reset streaming state
          setStreamingAssistantMessage(null);
          accumulatedContentRef.current = "";
          currentAssistantMessageId.current = null;
          break;

        // --- Handle backend confirmation/updates ---
        case "title_updated": // Handle confirmation that title was saved by backend
          console.log("Received title_updated confirmation:", message.payload);
          if (message.payload.chatId === chatId) {
            console.log("Refreshing history after title update confirmation.");
            refreshHistory(); // Refresh the sidebar list
          }
          break;
        case "message_saved": // Confirmation that user/assistant message saved (optional)
            console.log(`Message saved confirmation received: ID ${message.payload.id}`);
            // Could update optimistic message ID here if needed
            break;
        case "message_updated": // Confirmation that message edit saved
            console.log(`Message updated confirmation received: ID ${message.payload.id}`);
             refreshHistory(); // Refresh sidebar if needed after edit
            break;
        case "message_save_error":
            console.error("Backend error saving message:", message.payload.error);
            setError(`Failed to save message: ${message.payload.error}`);
            // Mark relevant message with error in UI?
            break;
        case "message_update_error":
            console.error("Backend error updating message:", message.payload.error);
            setError(`Failed to update message: ${message.payload.error}`);
            // Mark relevant message with error in UI?
            break;
        case "stopped": // Confirmation that generation was stopped by backend
            console.log("Backend confirmed generation stopped.");
            // UI state (isGenerating) is already handled in handleStopGeneration
            break;

        // --- Handle errors from backend ---
        case "error":
          console.error("WebSocket Server Error:", message.payload.error);
          setError(`Assistant error: ${message.payload.error}`);
          setIsGenerating(false);
          setStreamingAssistantMessage(null); // Clear any streaming state
          accumulatedContentRef.current = "";
          currentAssistantMessageId.current = null;
          break;
        default:
          console.warn("Unknown WebSocket message type:", message.type);
      }
    } catch (error) {
      console.error("Failed to parse WebSocket message or handle:", error, data);
      setError("Received invalid data from server.");
      // Reset potentially problematic state
      setIsGenerating(false);
      setStreamingAssistantMessage(null);
      accumulatedContentRef.current = "";
      currentAssistantMessageId.current = null;
      setIsLoadingHistory(false); // Stop loading if error occurs during history load
    }
  }, [chatId, refreshHistory, messages, streamingAssistantMessage]); // Added dependencies


  // --- Send Message / Regenerate Handler (WebSocket) ---
  const sendMessageOrRegenerate = async (
    contentToSend,
    isRegeneration = false
  ) => {
    if (!chatId) {
      setError("Cannot send message: No active chat selected.");
      console.error("sendMessageOrRegenerate called without chatId");
      return;
    }
    // Check WebSocket connection before proceeding
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      setError("WebSocket is not connected. Cannot send message.");
      console.error("WebSocket is not open. ReadyState:", ws.current?.readyState);
      // Optionally try to reconnect here
      return;
    }
    if (!contentToSend.trim() || (isGenerating && !isRegeneration)) return; // Allow stopping regen?
    setError(null);

    let messagesForSending = [...messages]; // Copy current valid messages (exclude streaming)

    // --- 1. Prepare User Message (if not regenerating) ---
    let userMessageId = null;
    if (!isRegeneration) {
      setInputValue(""); // Clear input only for new messages
      const optimisticUserMessage = {
        sender: "user",
        content: contentToSend,
        chat_id: chatId,
        user_id: user?.id, // Include user ID if available
        created_at: new Date().toISOString(),
        id: `temp-user-${uuidv4()}`, // Optimistic temporary ID
      };
      userMessageId = optimisticUserMessage.id;
      // Add user message optimistically to UI
      messagesForSending = [...messagesForSending, optimisticUserMessage];
      setMessages(messagesForSending); // Update UI immediately
      setShowPlaceholder(false); // Hide placeholder on first send

      // NOTE: Backend will save the user message based on the history sent in the 'chat' payload.
      // No separate "save_user_message" WS call needed here if backend handles it.
      if (messages.length === 0) {
        console.log("First user message, refreshing history later for title.");
        // refreshHistory(); // Refresh potentially after first assistant response instead
      }
    } else {
      // If regenerating, remove the last assistant message visually for context
      const lastAssistantMsgIndex = messagesForSending.findLastIndex(
        (msg) => msg.sender === "assistant"
      );
      if (lastAssistantMsgIndex > -1) {
        messagesForSending.splice(lastAssistantMsgIndex, 1);
        // No need to update main 'messages' state here yet,
        // as we'll replace it upon receiving the regenerated response.
        // setMessages(messagesForSending); // Don't update UI state yet
      }
    }

    // --- 2. Send "chat" Message via WebSocket ---
    try {
      const messagePayload = {
        // Send history *up to the point of the message being sent/regenerated*
        // Exclude any temporary/loading messages for backend processing
        messages: messagesForSending
          .filter(m => !m.isLoading && !m.id?.startsWith('temp-')) // Filter out loading/temp
          .slice(-20) // Limit history length sent to backend
          .map((m) => ({ sender: m.sender, content: m.content })), // Only send needed fields
        model: selectedModel || "default", // Ensure a model name is sent
        chatId: chatId,
        userId: user?.id, // Include the user ID
        // Include the first user message content *only if* this is the very first user message
        firstUserMessageContent:
          messages.filter((m) => m.sender === "user").length === 0 && !isRegeneration
            ? contentToSend
            : undefined, // Backend uses presence of this field
      };

      // If regenerating, ensure the *last* message in the payload is the user message
      if (isRegeneration) {
          const lastUserMsg = messagesForSending.filter(m => !m.isLoading && !m.id?.startsWith('temp-')).findLast(m => m.sender === 'user');
          if(lastUserMsg && messagePayload.messages[messagePayload.messages.length - 1]?.content !== lastUserMsg.content) {
             // This logic might need refinement depending on exact history structure desired for regen
             console.warn("Regeneration history payload might not end with the correct user message.");
          }
      }


      console.log("Sending WS 'chat' message:", messagePayload);
      ws.current.send(JSON.stringify({ type: "chat", payload: messagePayload }));
      setIsGenerating(true);

      // --- 3. Prepare Streaming State ---
      accumulatedContentRef.current = "";
      currentAssistantMessageId.current = `temp-assistant-${uuidv4()}`; // New temp ID
      setStreamingAssistantMessage({
        id: currentAssistantMessageId.current,
        sender: "assistant",
        content: "",
        chat_id: chatId,
        created_at: new Date().toISOString(),
        isLoading: true,
      });

    } catch (sendError) {
      console.error("WebSocket send error:", sendError);
      setError("Failed to communicate with the assistant.");
      setIsGenerating(false);
      setStreamingAssistantMessage(null);
      // Revert optimistic user message if send failed?
       if (!isRegeneration && userMessageId) {
         setMessages(prev => prev.filter(m => m.id !== userMessageId));
       }
    }
  };

  // --- Edit Message Handler ---
  const handleEditSave = async (messageId, newContent) => {
    if (isGenerating) return; // Don't allow edits while generating

    // Check WebSocket connection
     if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      setError("WebSocket is not connected. Cannot save edit.");
      return;
    }

    setError(null);
    const editedMessageIndex = messages.findIndex((msg) => msg.id === messageId);

    if (editedMessageIndex === -1 || messages[editedMessageIndex].sender !== "user") {
      setError("Only existing user messages can be edited.");
      return;
    }

    // 1. Prepare the updated message and the history up to that point
    const updatedMessage = { ...messages[editedMessageIndex], content: newContent };
    const historyUpToEdit = messages.slice(0, editedMessageIndex);
    const messagesForRegeneration = [...historyUpToEdit, updatedMessage];

    // 2. Update UI State: Show only messages up to the edited one
    setMessages(messagesForRegeneration);
    setShowPlaceholder(false);

    // 3. Send WebSocket message to backend to update DB and trigger regeneration
    try {
        const editPayload = {
            messageId: messageId,
            newContent: newContent,
            chatId: chatId,
            userId: user?.id,
             // Provide history for regeneration context
            historyForRegen: messagesForRegeneration
                .filter(m => !m.isLoading && !m.id?.startsWith('temp-'))
                .slice(-20)
                .map(m => ({ sender: m.sender, content: m.content })),
            model: selectedModel || "default",
        };
        console.log("Sending WS 'edit_message' message:", editPayload);
        ws.current.send(JSON.stringify({ type: "edit_message", payload: editPayload }));
        setIsGenerating(true); // Expecting regeneration stream

        // Prepare streaming state
        accumulatedContentRef.current = "";
        currentAssistantMessageId.current = `temp-assistant-${uuidv4()}`; // New temp ID
        setStreamingAssistantMessage({
            id: currentAssistantMessageId.current,
            sender: "assistant",
            content: "",
            chat_id: chatId,
            created_at: new Date().toISOString(),
            isLoading: true,
        });

    } catch (sendError) {
        console.error("WebSocket send error during edit:", sendError);
        setError("Failed to send edit request to server.");
        setIsGenerating(false);
        setStreamingAssistantMessage(null);
        // Consider reverting UI state on send error
        setMessages(messages); // Revert to original messages list
    }
  };
  // --- End Edit Message Handler ---

  // Specific handler for the form submission
  const handleSendMessage = (text) => {
    sendMessageOrRegenerate(text, false);
  };

  // Specific handler for the regenerate button
  const handleRegenerate = () => {
    if (isGenerating || !messages.length) return;
    // Find the last *user* message to use as context trigger
    const lastUserMessage = messages
        .filter(m => !m.isLoading && !m.id?.startsWith('temp-')) // Use only saved messages
        .slice()
        .reverse()
        .find((msg) => msg.sender === "user");

    if (lastUserMessage) {
      // Pass the content of the last user message to trigger regen
      // The function will handle slicing history correctly internally
      sendMessageOrRegenerate(lastUserMessage.content, true);
    } else {
      setError("Cannot regenerate: No previous user message found.");
    }
  };

  // Handler for copying text
  const handleCopy = async (text, messageId) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopySuccess(messageId);
      setTimeout(() => setCopySuccess(""), 1500);
    } catch (err) {
      console.error("Failed to copy text: ", err);
      setError("Failed to copy text.");
    }
  };

  // Handler for stopping generation
  const handleStopGeneration = () => {
    if (!isGenerating) return;
    console.log("Stopping generation...");
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "stop" }));
      console.log("Sent stop request to backend.");
    } else {
        console.warn("Cannot send stop request: WebSocket not connected.");
    }
    setIsGenerating(false); // Immediately update UI

    // Handle partially streamed message
    if (streamingAssistantMessage && accumulatedContentRef.current && currentAssistantMessageId.current) {
      const finalPartialMessage = {
        ...streamingAssistantMessage,
        content: accumulatedContentRef.current, // Use accumulated content
        isLoading: false,
      };
      // Add the partial message to the main list
      setMessages((prev) => {
          const existing = prev.find(msg => msg.id === finalPartialMessage.id);
          if (existing) {
              return prev.map(msg => msg.id === finalPartialMessage.id ? finalPartialMessage : msg);
          } else {
              return [...prev, finalPartialMessage];
          }
      });
       // NOTE: Backend needs to handle saving the partial message if desired upon receiving 'stop'
    }

    // Reset streaming state
    setStreamingAssistantMessage(null);
    accumulatedContentRef.current = "";
    currentAssistantMessageId.current = null;
  };

  const handleExampleClick = (text) => {
    setInputValue(text);
  };

  // Find the index of the last completed assistant message
  const lastAssistantMessageIndex = messages
      .filter(m => !m.isLoading && !m.id?.startsWith('temp-')) // Consider only saved messages
      .findLastIndex((msg) => msg.sender === "assistant");

  return (
    <>
      {(!showPlaceholder || messages.length > 0) && ( // Show model if not placeholder or if messages exist
        <div className="chat-model-selector">
          <span>{selectedModel || 'Default Model'}</span>
        </div>
      )}

      <div id="feed" className={showPlaceholder && !isLoadingHistory ? "welcome-screen" : ""}>
        {/* Show loading indicator */}
        {isLoadingHistory && (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-secondary)' }}>
                Loading chat history...
            </div>
        )}

        {/* Show chat feed only when not loading history */}
        {!isLoadingHistory && (
            <ChatFeed
              messages={messages}
              streamingMessage={streamingAssistantMessage}
              showPlaceholder={showPlaceholder}
              isGenerating={isGenerating}
              onExampleClick={handleExampleClick}
              bottomRef={bottomRef}
              assistantName={systemInfo?.assistantName || "Assistant"} // Use systemInfo safely
              onCopy={handleCopy}
              onRegenerate={handleRegenerate}
              onEditSave={handleEditSave}
              copySuccessId={copySuccess}
              lastAssistantMessageIndex={lastAssistantMessageIndex}
              // Add error display within messages if needed
            />
        )}

        {/* Show global error message */}
        {error && <div className="error-message chat-error">{error}</div>}

        {/* Input area */}
        <InputArea
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSendMessage}
          onStop={handleStopGeneration}
          isGenerating={isGenerating}
          disabled={isLoadingHistory || !isConnected.current} // Disable input while loading history or disconnected
        />
      </div>
    </>
  );
}

export default ChatPage;