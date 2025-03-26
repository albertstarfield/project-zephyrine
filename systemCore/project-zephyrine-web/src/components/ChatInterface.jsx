import React, { useState, useEffect, useRef, useCallback } from "react"; // Added useCallback
import { supabase } from "../utils/supabaseClient"; // Import Supabase client
import "../styles/ChatInterface.css";
import { ChevronDown, ChevronUp } from 'lucide-react'; // Example icons

// Define outside component or in a config file
const WEBSOCKET_URL = "ws://localhost:3001"; // Assuming backend runs on port 3001

const ChatInterface = ({ selectedModel = "llama3-8b-8192" }) => { // Added selectedModel prop with default
  const [messages, setMessages] = useState([]); // Initialize with empty array
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null); // Add error state
  const [isHistoryCollapsed, setIsHistoryCollapsed] = useState(false); // State for collapsing history
  const messagesEndRef = useRef(null); // Ref to scroll to bottom
  const ws = useRef(null); // Ref for WebSocket instance
  const currentAssistantMessageId = useRef(null); // Ref to track the ID of the message being streamed

  // Fetch messages on component mount
  useEffect(() => {
    const fetchMessages = async () => {
      const { data, error } = await supabase
        .from("messages")
        .select("*")
        .order("created_at", { ascending: true });

      if (error) {
        console.error("Error fetching messages:", error);
        setError("Failed to load messages.");
        setMessages([]); // Set to empty on error
      } else {
        setMessages(data || []); // Ensure data is not null
        setError(null);
      }
    };

    fetchMessages();

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
        // Attempt to reconnect after a delay
        // setTimeout(connectWebSocket, 5000); // Simple reconnect logic
      };

      ws.current.onclose = () => {
        console.log("WebSocket Disconnected");
        // Optionally attempt to reconnect or notify user
        // setError("WebSocket connection closed. Attempting to reconnect...");
        // setTimeout(connectWebSocket, 5000); // Simple reconnect logic
      };
    };

    connectWebSocket();
    // --- End WebSocket Setup ---


    // Optional: Set up real-time subscription (Keep commented out if using WebSocket for updates)
    // const subscription = supabase
    //   .channel('public:messages')
    //   .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'messages' }, payload => {
    //     setMessages(prevMessages => [...prevMessages, payload.new]);
    //   })
    //   .subscribe();

    // Cleanup WebSocket on unmount
    return () => {
      ws.current?.close();
      // supabase.removeChannel(subscription); // If using Supabase subscription
    };

  }, []); // Empty dependency array ensures this runs only once on mount

  // Scroll to bottom when messages change or history is expanded
  useEffect(() => {
    if (!isHistoryCollapsed) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isHistoryCollapsed]);

  // --- WebSocket Message Handler ---
  const handleWebSocketMessage = useCallback((data) => {
    try {
      const message = JSON.parse(data);
      // console.log("WS Message Received:", message); // Debug log

      switch (message.type) {
        case 'chunk':
          setIsGenerating(true); // Ensure generating state is true while receiving chunks
          setMessages((prevMessages) => {
            const lastMessage = prevMessages[prevMessages.length - 1];
            // If the last message is from the assistant and matches the current stream ID, append content
            if (lastMessage?.sender === 'assistant' && lastMessage.id === currentAssistantMessageId.current) {
              return prevMessages.map((msg) =>
                msg.id === currentAssistantMessageId.current
                  ? { ...msg, content: msg.content + message.payload.content }
                  : msg
              );
            } else {
              // Otherwise, add a new assistant message placeholder
              const newAssistantMessage = {
                // Use a temporary ID until saved to DB
                id: `temp-${Date.now()}`, // Or use a UUID library
                sender: 'assistant',
                content: message.payload.content,
                created_at: new Date().toISOString(),
              };
              currentAssistantMessageId.current = newAssistantMessage.id; // Track the new message ID
              return [...prevMessages, newAssistantMessage];
            }
          });
          break;
        case 'end':
          setIsGenerating(false);
          // Save the completed message to Supabase
          const finalMessage = messages.find(msg => msg.id === currentAssistantMessageId.current);
          if (finalMessage) {
             saveAssistantMessage(finalMessage.content);
          }
          currentAssistantMessageId.current = null; // Reset tracker
          break;
        case 'error':
          console.error("WebSocket Server Error:", message.payload.error);
          setError(`Assistant error: ${message.payload.error}`);
          setIsGenerating(false);
          currentAssistantMessageId.current = null; // Reset tracker
          break;
        default:
          console.warn("Unknown WebSocket message type:", message.type);
      }
    } catch (error) {
      console.error("Failed to parse WebSocket message:", error);
      setError("Received invalid data from server.");
      setIsGenerating(false); // Stop generation on parse error
      currentAssistantMessageId.current = null; // Reset tracker
    }
  }, [messages]); // Include messages in dependency array for saving final message


  // --- Function to save assistant message to Supabase ---
  const saveAssistantMessage = async (content) => {
    const { data: insertedAssistantMessage, error: assistantInsertError } = await supabase
      .from("messages")
      .insert([{ sender: "assistant", content: content }])
      .select()
      .single();

    if (assistantInsertError) {
      console.error("Error saving assistant message:", assistantInsertError);
      setError("Failed to save assistant response.");
      // Optionally handle UI update if needed (e.g., mark message as unsaved)
    } else if (insertedAssistantMessage) {
      // Replace the temporary message ID with the actual ID from Supabase
      setMessages(prevMessages => prevMessages.map(msg =>
        msg.id === currentAssistantMessageId.current ? { ...msg, id: insertedAssistantMessage.id } : msg
      ));
    }
  };


  // --- Send Message Handler ---
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isGenerating) return; // Prevent sending while generating
    setError(null); // Clear previous errors

    const userMessageContent = inputValue;
    setInputValue(""); // Clear input immediately

    // 1. Save User Message to Supabase
    const { data: insertedMessage, error: insertError } = await supabase
      .from("messages")
      .insert([{ sender: "user", content: userMessageContent }])
      .select()
      .single();

    if (insertError) {
      console.error("Error sending message:", insertError);
      setError("Failed to send message.");
      setInputValue(userMessageContent); // Restore input value
      return;
    }

    // 2. Update Local State with User Message
    let currentMessages = [];
    if (insertedMessage) {
       setMessages((prev) => {
           currentMessages = [...prev, insertedMessage];
           return currentMessages;
       });
    } else {
        // Fallback if insert didn't return data (should not happen with .select())
        currentMessages = [...messages, { id: `temp-user-${Date.now()}`, sender: 'user', content: userMessageContent, created_at: new Date().toISOString() }];
        setMessages(currentMessages);
    }


    // 3. Send Message via WebSocket
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      try {
        const messageToSend = {
          type: 'chat',
          payload: {
            // Send relevant message history (adjust as needed for context)
            messages: currentMessages.slice(-10), // Example: send last 10 messages
            model: selectedModel // Use the model passed via props
          }
        };
        ws.current.send(JSON.stringify(messageToSend));
        setIsGenerating(true); // Start generating indicator
        currentAssistantMessageId.current = null; // Reset assistant message tracker for the new request
      } catch (sendError) {
        console.error("WebSocket send error:", sendError);
        setError("Failed to communicate with the assistant.");
        setIsGenerating(false);
      }
    } else {
      setError("WebSocket is not connected. Cannot send message.");
      console.error("WebSocket is not open. ReadyState:", ws.current?.readyState);
      // Optionally try to reconnect here
      // connectWebSocket();
    }
  };

  const toggleHistoryCollapse = () => {
    setIsHistoryCollapsed(!isHistoryCollapsed);
  };

  // Keep handleStopGeneration if you plan to implement actual async generation later
  const handleStopGeneration = () => {
    console.log("Stopping generation (placeholder)");
    setIsGenerating(false);
    // Add logic here to cancel any ongoing backend request if applicable
  };

  return (
    <div className="chat-interface">
       {/* Add a header for the toggle button */}
       <div className="chat-history-header">
         <button onClick={toggleHistoryCollapse} className="collapse-toggle-button">
           {isHistoryCollapsed ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
           <span>{isHistoryCollapsed ? 'Show' : 'Hide'} History</span>
         </button>
       </div>

      {/* Apply conditional class and add a ref container */}
      <div className={`chat-messages-container ${isHistoryCollapsed ? 'collapsed' : 'expanded'}`}>
        <div className="chat-messages">
          {/* Display error message if any */}
          {error && <div className="error-message">{error}</div>}

        {/* Render messages */}
        {messages.map((message) => (
          <div
            // Use the actual ID from Supabase (or optimisticId temporarily)
            key={message.id}
            className={`message ${
              message.sender === "user" ? "user-message" : "assistant-message"
            }`}
          >
            <div className="message-content">{message.content}</div>
            {/* Optional: Display timestamp */}
            {/* <div className="message-timestamp">
              {new Date(message.created_at).toLocaleTimeString()}
            </div> */}
          </div>
        ))}
        {/* Add a div to help scroll to bottom */}
        <div ref={messagesEndRef} />
        {/* Keep generating indicator if needed later */}
        {isGenerating && (
          <div className="generating-indicator">
            <span>Generating response</span>
            <span className="dot-animation">...</span>
          </div>
        )}
        </div> {/* End chat-messages */}
      </div> {/* End chat-messages-container */}

      <form className="chat-input-area" onSubmit={handleSendMessage}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type your message here..."
          disabled={isGenerating}
        />
        {isGenerating ? (
          <button
            type="button"
            className="stop-button"
            onClick={handleStopGeneration}
          >
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!inputValue.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  );
};

export default ChatInterface;
