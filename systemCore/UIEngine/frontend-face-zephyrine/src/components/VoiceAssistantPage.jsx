// externalAnalyzer/frontend-face-zephyrine/src/components/VoiceAssistantPage.jsx

import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import '../styles/components/_voiceAssistantPage.css';
import { Mic, MicOff, Send } from 'lucide-react'; // Example icons, you can use any library

const VoiceAssistantPage = () => {
  const { user } = useAuth();
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [messages, setMessages] = useState([]); // To hold the conversation history

  // Set a welcome message when the component mounts or the user changes
  useEffect(() => {
    setMessages([
      {
        id: 'welcome-1',
        sender: 'assistant',
        text: `Hello, ${user?.name || 'there'}! I'm Zephyrine. How can I assist you today?`,
        timestamp: new Date(),
      },
    ]);
  }, [user]);

  // Placeholder function to handle sending a message
  const handleSendMessage = () => {
    if (!transcript.trim()) return;

    const newUserMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: transcript,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, newUserMessage]);
    setTranscript(''); // Clear the input after sending

    // Simulate the assistant's response
    setTimeout(() => {
      const assistantResponse = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        text: `This is a placeholder response for: "${transcript}"`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantResponse]);
    }, 1000);
  };

  // Placeholder function to toggle voice recognition
  const toggleListening = () => {
    setIsListening(prev => !prev);
    // In a real implementation, you would integrate the Web Speech API here.
    if (!isListening) {
      setTranscript("Listening... (this is a placeholder)");
    } else {
      setTranscript(""); // Clear when stopping
    }
  };

  return (
    <div className="voice-assistant-container">
      {/* Renewed logo as requested */}
      <img src="/img/ProjectZephy023LogoRenewal.png" alt="Project Zephyrine Logo" className="assistant-page-logo" />

      <div className="assistant-header">
        <h1>Voice Assistant</h1>
        <p>Speak your commands or questions directly to Zephyrine.</p>
      </div>

      <div className="conversation-window">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.sender}`}>
            <p className="message-text">{message.text}</p>
            <span className="message-timestamp">
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        ))}
      </div>

      <div className="input-area">
        <input
          type="text"
          className="text-input"
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          placeholder={isListening ? 'Listening...' : 'Type your message or use the mic...'}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
        />
        <button
          className={`mic-button ${isListening ? 'listening' : ''}`}
          onClick={toggleListening}
        >
          {isListening ? <MicOff size={24} /> : <Mic size={24} />}
        </button>
        <button
            className="send-button"
            onClick={handleSendMessage}
            disabled={!transcript.trim()}
        >
            <Send size={24} />
        </button>
      </div>
    </div>
  );
};

export default VoiceAssistantPage;