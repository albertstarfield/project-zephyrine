import React, { useState, useEffect } from "react";
import { supabase } from "../utils/supabaseClient"; // Import Supabase client
import "../styles/ChatInterface.css";

const ChatInterface = () => {
  const [messages, setMessages] = useState([]); // Initialize with empty array
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false); // Keep this for potential future use with actual generation
  const [error, setError] = useState(null); // Add error state

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

  // Keep handleStopGeneration if you plan to implement actual async generation later
  const handleStopGeneration = () => {
    console.log("Stopping generation (placeholder)");
    setIsGenerating(false);
    // Add logic here to cancel any ongoing backend request if applicable
  };

  return (
    <div className="chat-interface">
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
        {/* Keep generating indicator if needed later */}
        {isGenerating && (
          <div className="generating-indicator">
            <span>Generating response</span>
            <span className="dot-animation">...</span>
          </div>
        )}
      </div>

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
