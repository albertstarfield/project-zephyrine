import React, { useState } from "react";
import "../styles/ChatInterface.css";

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: "assistant",
      content:
        "Hello! I am Adelaide Zephyrine Charlotte. How can I assist you today?",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  const handleSendMessage = (e) => {
    e.preventDefault();

    if (!inputValue.trim()) return;

    // Add user message
    const userMessage = {
      id: messages.length + 1,
      sender: "user",
      content: inputValue,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");

    // Simulate assistant response
    setIsGenerating(true);

    setTimeout(() => {
      const assistantMessage = {
        id: messages.length + 2,
        sender: "assistant",
        content:
          "This is a simulated response. In a real implementation, this would connect to a backend service.",
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setIsGenerating(false);
    }, 1500);
  };

  const handleStopGeneration = () => {
    setIsGenerating(false);
  };

  return (
    <div className="chat-interface">
      <div className="chat-messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${
              message.sender === "user" ? "user-message" : "assistant-message"
            }`}
          >
            <div className="message-content">{message.content}</div>
          </div>
        ))}
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
