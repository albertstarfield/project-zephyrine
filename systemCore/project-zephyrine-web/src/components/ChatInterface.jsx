import React, { useState, useEffect, useRef } from "react";
import { supabase } from "../utils/supabaseClient"; // Import Supabase client
import "../styles/ChatInterface.css";
import { ChevronDown, ChevronUp } from 'lucide-react'; // Example icons

const ChatInterface = () => {
  const [messages, setMessages] = useState([]); // Initialize with empty array
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false); // Keep this for potential future use with actual generation
  const [error, setError] = useState(null); // Add error state
  const [isHistoryCollapsed, setIsHistoryCollapsed] = useState(false); // State for collapsing history
  const messagesEndRef = useRef(null); // Ref to scroll to bottom

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

    // Optional: Set up real-time subscription
    // const subscription = supabase
    //   .channel('public:messages')
    //   .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'messages' }, payload => {
    //     setMessages(prevMessages => [...prevMessages, payload.new]);
    //   })
    //   .subscribe();

    // // Cleanup subscription on unmount
    // return () => {
    //   supabase.removeChannel(subscription);
    // };

  }, []);

  // Scroll to bottom when messages change or history is expanded
  useEffect(() => {
    if (!isHistoryCollapsed) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isHistoryCollapsed]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    setError(null); // Clear previous errors

    if (!inputValue.trim()) return;

    const userMessageContent = inputValue;
    setInputValue(""); // Clear input immediately

    // Optimistically add user message to UI (optional, improves perceived speed)
    // const optimisticId = Date.now(); // Temporary ID
    // setMessages((prev) => [
    //   ...prev,
    //   { id: optimisticId, sender: "user", content: userMessageContent, created_at: new Date().toISOString() },
    // ]);

    // Insert user message into Supabase
    const { data: insertedMessage, error: insertError } = await supabase
      .from("messages")
      .insert([{ sender: "user", content: userMessageContent }])
      .select() // Return the inserted row
      .single(); // Expecting a single row back

    if (insertError) {
      console.error("Error sending message:", insertError);
      setError("Failed to send message.");
      // Optional: Remove optimistic message if insertion failed
      // setMessages((prev) => prev.filter(msg => msg.id !== optimisticId));
      setInputValue(userMessageContent); // Restore input value
      return;
    }

    // Update state with the actual message from Supabase (if not using optimistic update or subscription)
    if (insertedMessage) {
       setMessages((prev) => [...prev, insertedMessage]);
    }


    // --- Assistant Response Logic Placeholder ---
    // Here you would typically call your backend/AI service
    // For now, we are just saving the user message.
    // setIsGenerating(true);
    // const assistantResponse = await getAssistantResponse(userMessageContent); // Example function
    // if (assistantResponse) {
    //   const { data: insertedAssistantMessage, error: assistantInsertError } = await supabase
    //     .from("messages")
    //     .insert([{ sender: "assistant", content: assistantResponse }])
    //     .select()
    //     .single();
    //   if (insertedAssistantMessage) {
    //      setMessages((prev) => [...prev, insertedAssistantMessage]);
    //   } else {
    //      console.error("Error saving assistant message:", assistantInsertError);
    //      setError("Failed to save assistant response.");
    //   }
    // } else {
    //    setError("Failed to get assistant response.");
    // }
    // setIsGenerating(false);
    // --- End Placeholder ---
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
