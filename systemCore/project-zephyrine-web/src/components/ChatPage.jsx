import { useState, useEffect, useRef, useCallback } from 'react'; // Added useCallback
import { useParams } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import { supabase } from '../utils/supabaseClient';
import ChatFeed from "./ChatFeed";
import InputArea from './InputArea';
import '../styles/ChatInterface.css'; // Keep relevant styles if needed
import '../styles/utils/_overlay.css'; // Keep relevant styles if needed

// Define WebSocket URL
const WEBSOCKET_URL = "ws://localhost:3001";

// Component to handle individual chat sessions
function ChatPage({ systemInfo, user, refreshHistory, selectedModel }) {
  const { chatId } = useParams();
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showPlaceholder, setShowPlaceholder] = useState(true);
  const [error, setError] = useState(null);
  const [streamingAssistantMessage, setStreamingAssistantMessage] = useState(null); // State for the message being streamed
  const bottomRef = useRef(null);
  const ws = useRef(null); // Ref for WebSocket instance
  const currentAssistantMessageId = useRef(null); // Ref to track the ID of the message being streamed
  const accumulatedContentRef = useRef(""); // Ref to accumulate content chunks

  // Fetch messages when chatId changes
  useEffect(() => {
    const fetchMessages = async () => {
      if (!chatId) return; // Don't fetch if chatId isn't available yet

      setError(null); // Clear previous errors
      setIsGenerating(true); // Show loading state while fetching

      const { data, error: fetchError } = await supabase
        .from("messages")
        .select("*")
        .eq("chat_id", chatId) // Filter by the current chat ID
        .order("created_at", { ascending: true });

      if (fetchError) {
        console.error("Error fetching messages:", fetchError);
        setError("Failed to load chat history.");
        setMessages([]); // Clear messages on error
      } else {
        setMessages(data || []); // Ensure data is not null
        setShowPlaceholder(data === null || data.length === 0); // Show placeholder only if no messages
      }
      setIsGenerating(false); // Hide loading state
    };

    fetchMessages();
    setInputValue(""); // Clear input when changing chats

    // Optional: Set up real-time subscription for this specific chat
    // ... (subscription code commented out as in original)

    // // Cleanup subscription on unmount or chatId change
    // return () => {
    //   supabase.removeChannel(subscription);
    // };

    // --- WebSocket Connection Setup ---
    const connectWebSocket = () => {
      console.log("Attempting to connect WebSocket...");
      ws.current = new WebSocket(WEBSOCKET_URL);

      ws.current.onopen = () => {
        console.log("WebSocket Connected");
        setError(null); // Clear connection errors on successful connect
      };

      ws.current.onmessage = (event) => {
        handleWebSocketMessage(event.data);
      };

      ws.current.onerror = (err) => {
        console.error("WebSocket Error:", err);
        setError("WebSocket connection error. Please try refreshing.");
        // Consider adding reconnect logic here if needed
      };

      ws.current.onclose = () => {
        console.log("WebSocket Disconnected");
        // Consider adding reconnect logic or user notification
      };
    };

    connectWebSocket();
    // --- End WebSocket Setup ---

    // Cleanup WebSocket on unmount or chatId change
    return () => {
      ws.current?.close();
      // supabase.removeChannel(subscription); // If using Supabase subscription
    };

  }, [chatId]); // Re-run effect when chatId changes

  // Scroll to bottom when messages change
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streamingAssistantMessage]); // Also scroll when streaming message updates


  // --- WebSocket Message Handler ---
  const handleWebSocketMessage = useCallback((data) => {
    try {
      const message = JSON.parse(data);
      // console.log("WS Message Received:", message); // Debug log

      switch (message.type) {
        case 'chunk':
          setIsGenerating(true);
          const contentChunk = message.payload.content;
          accumulatedContentRef.current += contentChunk; // Append to ref

          // Update UI state for streaming display
          setStreamingAssistantMessage(prev => {
            // Ensure we have a streaming message object to update
            if (!prev && currentAssistantMessageId.current) {
              // Initialize if it's the first chunk for this message ID
              return {
                id: currentAssistantMessageId.current,
                sender: 'assistant',
                content: accumulatedContentRef.current, // Use accumulated content
                chat_id: chatId,
                created_at: new Date().toISOString(), // Consider using a fixed start time
                isLoading: true,
              };
            } else if (prev) {
               // Update existing streaming message content
               return { ...prev, content: accumulatedContentRef.current };
            }
            return prev; // Should not happen if currentAssistantMessageId is set
          });
          break;
        case 'end':
          setIsGenerating(false);
          const finalContent = accumulatedContentRef.current; // Get final content from ref

          if (finalContent && currentAssistantMessageId.current) {
            // Create the final message object
            const finalMessage = {
              ...streamingAssistantMessage, // Get base details like chatId, created_at
              id: currentAssistantMessageId.current, // Use the tracked ID
              content: finalContent,
              isLoading: false,
            };

            // Save to Supabase
            saveAssistantMessage(finalContent, finalMessage.id); // Pass ID for potential update

            // Add the final message to the main messages list
            setMessages(prev => [...prev, finalMessage]);
          } else {
             console.log("End event received but no content accumulated or ID tracked.");
          }

          // Reset for next message
          setStreamingAssistantMessage(null);
          accumulatedContentRef.current = "";
          currentAssistantMessageId.current = null;
          break;
        case 'error':
          console.error("WebSocket Server Error:", message.payload.error);
          setError(`Assistant error: ${message.payload.error}`);
          setIsGenerating(false);
          setStreamingAssistantMessage(null);
          accumulatedContentRef.current = "";
          currentAssistantMessageId.current = null;
          break;
        default:
          console.warn("Unknown WebSocket message type:", message.type);
      }
    } catch (error) {
      console.error("Failed to parse WebSocket message:", error);
      setError("Received invalid data from server.");
      setIsGenerating(false);
      setStreamingAssistantMessage(null);
      accumulatedContentRef.current = "";
      currentAssistantMessageId.current = null;
    }
  }, [chatId]); // Dependencies - Removed streamingAssistantMessage


  // --- Function to save assistant message to Supabase ---
  const saveAssistantMessage = async (content, tempId) => { // Accept tempId
    if (!content || !chatId) return;

    const { data: dbAssistantMessage, error: assistantSaveError } = await supabase
      .from('messages')
      .insert([
        {
          sender: 'assistant',
          content: content,
          chat_id: chatId,
          user_id: user?.id, // Associate with user if logged in
        },
      ])
      .select()
      .single();

    if (assistantSaveError) {
      console.error("Error saving assistant message:", assistantSaveError);
      setError("Failed to save assistant response.");
      // Update the message in the list to show an error state
      setMessages(prev => prev.map(msg =>
        msg.id === tempId ? { ...msg, error: 'Failed to save' } : msg
      ));
    } else if (dbAssistantMessage) {
      // Replace the temporary message with the final one from the database
      setMessages(prev => {
        const existingMessages = prev.filter(msg => msg.id !== tempId); // Remove potential temp message
        return [...existingMessages, dbAssistantMessage]; // Add the final message
      });
      console.log("Assistant message saved:", dbAssistantMessage.id);
    }
  };


  // --- Send Message Handler (WebSocket) ---
  const handleSendMessage = async (text) => {
    if (!text.trim() || !chatId || isGenerating) return; // Prevent sending while generating

    setError(null);
    const userMessageContent = text;
    setInputValue(''); // Clear input field
    setShowPlaceholder(false); // Hide placeholder on send

    // --- 1. Prepare and Save User Message ---
    const userMessageData = {
      sender: 'user',
      content: userMessageContent,
      chat_id: chatId,
      user_id: user?.id,
    };
    // Add user message optimistically to UI
    const optimisticUserMessage = { ...userMessageData, created_at: new Date().toISOString(), id: uuidv4() };
    let currentMessages = [...messages, optimisticUserMessage];
    setMessages(currentMessages);

    // Save user message to DB (no need to await)
    supabase.from('messages').insert([userMessageData]).select().single().then(({ data: dbUserMessage, error: insertError }) => {
      if (insertError) {
        console.error("Error saving user message:", insertError);
        setError("Failed to save your message.");
        setMessages(prev => prev.map(msg => msg.id === optimisticUserMessage.id ? { ...msg, error: 'Failed to save' } : msg));
      } else {
        // Optionally update message ID if needed, or just log success
        console.log("User message saved:", dbUserMessage?.id);
        // Refresh history if it was the first message
        if (messages.length === 0) refreshHistory();
      }
    });

    // --- 2. Send Message via WebSocket ---
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      try {
        const messageToSend = {
          type: 'chat',
          payload: {
            // Send relevant message history (adjust context length as needed)
            messages: currentMessages.slice(-10).map(m => ({ sender: m.sender, content: m.content })), // Format for backend
            model: selectedModel
          }
        };
        ws.current.send(JSON.stringify(messageToSend));
        setIsGenerating(true);

        // --- 3. Prepare Streaming State (Initialize refs and state) ---
        accumulatedContentRef.current = ""; // Reset accumulator
        currentAssistantMessageId.current = `temp-assistant-${Date.now()}`; // Generate temporary ID
        setStreamingAssistantMessage({ // Set initial streaming state for UI
          id: currentAssistantMessageId.current,
          sender: 'assistant',
          content: '', // Start empty
          chat_id: chatId,
          created_at: new Date().toISOString(),
          isLoading: true,
        });

      } catch (sendError) {
        console.error("WebSocket send error:", sendError);
        setError("Failed to communicate with the assistant.");
        setIsGenerating(false);
        setStreamingAssistantMessage(null); // Clear placeholder on send error
      }
    } else {
      setError("WebSocket is not connected. Cannot send message.");
      console.error("WebSocket is not open. ReadyState:", ws.current?.readyState);
      // Optionally try to reconnect here
    }
  };


  const handleStopGeneration = () => {
    // Basic stop: close WebSocket or send a 'stop' message if backend supports it
    console.log("Stopping generation...");
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
       // Option 1: Send a stop message (if backend handles it)
       // ws.current.send(JSON.stringify({ type: 'stop' }));

       // Option 2: Just close the connection (might be abrupt)
       // ws.current.close();
    }
    setIsGenerating(false);
    // Finalize potentially partial message if needed, or just clear it
    if (streamingAssistantMessage && streamingAssistantMessage.content) {
       saveAssistantMessage(streamingAssistantMessage.content); // Save what we have
       setMessages(prev => [...prev, { ...streamingAssistantMessage, isLoading: false }]);
    }
    setStreamingAssistantMessage(null);
    currentAssistantMessageId.current = null;
  };

  const handleExampleClick = (text) => {
    setInputValue(text);
    // Optionally focus the input area here
  };

  return (
    <>
      {/* Model Selector Display */}
      {!showPlaceholder && (
        <div className="chat-model-selector">
          <span>{selectedModel}</span>
        </div>
      )}

      {/* Main feed and input area */}
      <div id="feed" className={showPlaceholder ? "welcome-screen" : ""}>
        <ChatFeed
          messages={messages}
          streamingMessage={streamingAssistantMessage}
          showPlaceholder={showPlaceholder}
          isGenerating={isGenerating}
          onExampleClick={handleExampleClick}
          bottomRef={bottomRef}
          assistantName={systemInfo.assistantName}
        />
        {error && <div className="error-message chat-error">{error}</div>}
        <InputArea
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSendMessage}
          onStop={handleStopGeneration}
          isGenerating={isGenerating}
        />
      </div>
    </>
  );
}

export default ChatPage;
