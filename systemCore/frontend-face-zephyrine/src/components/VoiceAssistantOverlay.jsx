import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import { X } from 'lucide-react';
import ConversationTurn from './ConversationTurn';
import { useVoiceProcessor } from '../hooks/useVoiceProcessor';
import '../styles/components/_voiceOverlay.css';

const VoiceAssistantOverlay = ({ onExit }) => {
  const [conversationLog, setConversationLog] = useState([]);
  
  const { 
    voiceState,     // 'idle', 'listening', 'processing', 'speaking'
    error,          
    transcribedText, // Object { text: "...", final: true }
    start,          
    stop,           
  } = useVoiceProcessor();

  const logRef = useRef(null);

  // Effect to automatically start and stop the voice processor
  useEffect(() => {
    console.log("VoiceAssistantOverlay: Starting voice processor...");
    start();
    return () => {
      console.log("VoiceAssistantOverlay: Stopping voice processor...");
      stop();
    };
  }, [start, stop]); // These functions are stable from useCallback

  // Effect to update the conversation log when a final transcription arrives
  useEffect(() => {
    if (transcribedText && transcribedText.final) {
      console.log("Overlay received final transcription:", transcribedText.text);
      // Add user's turn to the conversation log
      setConversationLog(prevLog => [...prevLog, { speaker: 'user', text: transcribedText.text }]);
      
      // The hook's simulation will handle the "assistant" response for now
      // Awaiting real TTS response...
      // In the final version, the hook will provide the assistant's text too.
      // For now, let's just log it.
      setTimeout(() => {
        setConversationLog(prevLog => [...prevLog, { speaker: 'assistant', text: "This is a simulated response." }]);
      }, 2100); // Match hook's simulation timing
    }
  }, [transcribedText]);

  // Effect to scroll the conversation log down
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [conversationLog]);

  const getStatusText = () => {
    if (error) return error; 

    switch (voiceState) {
      case 'listening': return 'Listening...';
      case 'processing': return 'Thinking...';
      case 'speaking': return 'Speaking...';
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
        <div className="voice-conversation-log" ref={logRef}>
          {conversationLog.map((turn, index) => (
            <ConversationTurn 
              key={index} 
              userText={turn.speaker === 'user' ? turn.text : null}
              assistantText={turn.speaker === 'assistant' ? turn.text : null}
            />
          ))}
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