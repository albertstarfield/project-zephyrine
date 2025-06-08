// ExternalAnalyzer/frontend-face-zephyrine/src/components/VoiceAssistantOverlay.jsx
import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { X } from 'lucide-react';
import ConversationTurn from './ConversationTurn'; // Import the new component
import '../styles/components/_voiceOverlay.css';

// This component now manages the UI state of the voice interaction
const VoiceAssistantOverlay = ({ onExit }) => {
  // States for the voice interaction lifecycle: 'idle', 'listening', 'processing', 'speaking'
  const [voiceState, setVoiceState] = useState('idle'); 
  const [conversation, setConversation] = useState([]); // Array to hold conversation turns
  const [currentTranscription, setCurrentTranscription] = useState(''); // Live transcription text

  // This effect will be used later to start and stop the voice processor hook
  useEffect(() => {
    // In the next part, we'll call the voice processing hook here
    console.log("Voice Assistant Overlay Mounted. Current state:", voiceState);

    // Example of how we might simulate the conversation flow for UI testing:
    const simulateFlow = false; // Set to true to test UI flow
    if (simulateFlow) {
      setTimeout(() => setVoiceState('listening'), 2000);
      setTimeout(() => {
        setVoiceState('processing');
        setCurrentTranscription('Hello Zephyrine, how is the weather today?');
      }, 6000);
      setTimeout(() => {
        setVoiceState('speaking');
        setConversation([{ 
          user: 'Hello Zephyrine, how is the weather today?', 
          assistant: 'The weather in Jakarta is currently sunny with a high of 32 degrees Celsius.' 
        }]);
        setCurrentTranscription('');
      }, 9000);
      setTimeout(() => setVoiceState('idle'), 13000);
    }
  }, [voiceState]); // Re-run effect if voiceState changes (for cleanup/setup logic later)

  const getStatusText = () => {
    switch (voiceState) {
      case 'listening':
        return 'Listening...';
      case 'processing':
        return 'Thinking...';
      case 'speaking':
        return 'Speaking...';
      case 'idle':
      default:
        return 'Waiting for you to speak...';
    }
  };

  return (
    <div className="voice-overlay-container">
      <button className="voice-overlay-exit-button" onClick={onExit} title="Exit Voice Mode">
        <X size={32} />
      </button>
      
      <div className="voice-overlay-content">
        <div className="voice-conversation-log">
          {/* Render past conversation turns */}
          {conversation.map((turn, index) => (
            <ConversationTurn key={index} userText={turn.user} assistantText={turn.assistant} />
          ))}
          {/* Render live transcription */}
          {(voiceState === 'listening' || voiceState === 'processing') && currentTranscription && (
            <ConversationTurn userText={currentTranscription} />
          )}
        </div>

        <div className="voice-visualizer-container">
          <div className={`voice-visualizer-pulse ${voiceState === 'listening' ? 'active' : ''}`}></div>
          <div className={`voice-visualizer-pulse ${voiceState === 'listening' ? 'active' : ''}`} style={{animationDelay: '0.5s'}}></div>
          <div className={`voice-visualizer-pulse ${voiceState === 'listening' ? 'active' : ''}`} style={{animationDelay: '1s'}}></div>
        </div>
        
        <p className="voice-status-indicator">{getStatusText()}</p>
      </div>
    </div>
  );
};

VoiceAssistantOverlay.propTypes = {
  onExit: PropTypes.func.isRequired,
};

export default VoiceAssistantOverlay;