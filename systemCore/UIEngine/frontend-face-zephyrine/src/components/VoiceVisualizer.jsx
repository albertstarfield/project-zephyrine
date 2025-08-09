// src/components/VoiceVisualizer.jsx

import React from 'react';
import '../styles/components/_voiceVisualizer.css';

/**
 * A component to visualize the voice assistant's state.
 * @param {{
 *  status: 'idle' | 'listening' | 'processing' | 'speaking',
 *  amplitude: number
 * }} props
 */
const VoiceVisualizer = ({ status, amplitude }) => {
  // Define a base scale and apply the amplitude. Clamp to prevent zero-size.
  const getDotStyle = (ampMultiplier = 1) => {
    const scale = Math.max(0.1, 1 + amplitude * ampMultiplier);
    return { transform: `scale(${scale})` };
  };

  return (
    <div className="visualizer-container">
      <div className="dots-container">
        {/* The status class will now mainly control color */}
        <div className={`dots-wrapper ${status}`}>
          {/* Dots react differently for a more dynamic feel */}
          <div className="dot" style={getDotStyle(1.2)}></div>
          <div className="dot" style={getDotStyle(1.5)}></div>
          <div className="dot" style={getDotStyle(1.5)}></div>
          <div className="dot" style={getDotStyle(1.2)}></div>
        </div>
      </div>
      <div className="status-text">
        {status === 'listening' && "Listening..."}
        {status === 'processing' && "Thinking..."}
        {status === 'speaking' && "Speaking..."}
        {status === 'idle' && "Press the mic to start"}
      </div>
    </div>
  );
};

export default VoiceVisualizer;