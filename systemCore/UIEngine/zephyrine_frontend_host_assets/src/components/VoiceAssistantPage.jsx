// src/components/VoiceAssistantPage.jsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { MicVAD, utils } from '@ricky0123/vad-web';
import '../styles/components/_voiceAssistantPage.css';

import VoiceVisualizer from './VoiceVisualizer';

const VoiceAssistantPage = () => {
  // 'idle', 'listening', 'processing_stt', 'processing_llm', 'speaking'
  const [assistantStatus, setAssistantStatus] = useState('idle');
  const [amplitude, setAmplitude] = useState(0);
  const [error, setError] = useState(null);

  // Use refs to hold objects that should not trigger re-renders
  const vadRef = useRef(null);
  const animationFrameId = useRef(null);
  const chatHistory = useRef([
    { role: 'system', content: 'You are Zephyrine, a helpful voice assistant. Keep your responses concise.' }
  ]);

  // --- API Call Functions ---

  const transcribeAudio = async (audioBlob) => {
    setAssistantStatus('processing_stt'); // STT = Speech-to-Text
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.webm');
    formData.append('model', 'whisper-1'); // Or your model

    const response = await fetch('/api/v1/audio/transcriptions', {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error('Transcription failed');
    const data = await response.json();
    return data.text;
  };

  const getChatCompletion = async (text) => {
    setAssistantStatus('processing_llm'); // LLM is processing
    chatHistory.current.push({ role: 'user', content: text });

    const response = await fetch('/api/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'default', // Or your selected model
        messages: chatHistory.current,
      }),
    });
    if (!response.ok) throw new Error('Chat completion failed');
    const data = await response.json();
    const assistantResponse = data.choices[0].message.content;
    chatHistory.current.push({ role: 'assistant', content: assistantResponse });
    return assistantResponse;
  };

  const getSpeechAudio = async (text) => {
    setAssistantStatus('speaking');
    const response = await fetch('/api/v1/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'tts-1', // Or your model
        input: text,
        voice: 'alloy', // Or your desired voice
      }),
    });
    if (!response.ok) throw new Error('Speech synthesis failed');
    return response.blob();
  };

  const playAudio = (audioBlob) => {
    return new Promise((resolve) => {
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        resolve();
      };
    });
  };

  // --- Main VAD Lifecycle ---

  const startVad = useCallback(async () => {
    setError(null);
    try {
      const vad = await MicVAD.new({
        onSpeechEnd: async (audio) => {
          vad.pause(); // Pause VAD while processing
          const audioBlob = new Blob([audio], { type: 'audio/webm' });
          
          try {
            // The full conversation loop
            const userText = await transcribeAudio(audioBlob);
            if (userText.trim()) {
                const assistantText = await getChatCompletion(userText);
                const assistantAudio = await getSpeechAudio(assistantText);
                await playAudio(assistantAudio);
            }
          } catch (err) {
            console.error("Error in conversation loop:", err);
            setError(err.message);
            await new Promise(res => setTimeout(res, 2000)); // Show error for 2s
          }
          
          setAssistantStatus('idle'); // Return to idle, ready to listen
          vad.start(); // Resume VAD
        },
        // Continuously update amplitude for visualization
        onFrameProcessed: (data) => {
            if (data) {
                setAmplitude(data.some(v => v > 0.9) ? 1.0 : data.reduce((a, b) => a + b, 0) / data.length);
            }
        }
      });
      vad.start();
      vadRef.current = vad;
      setAssistantStatus('idle'); // Initial state ready to listen
    } catch (err) {
      console.error("VAD initialization failed:", err);
      setError("Microphone access denied. Please enable it in your browser settings.");
    }
  }, []);

  // Cleanup effect
  useEffect(() => {
    startVad(); // Start the process when the component mounts

    return () => {
      // This is crucial to release the microphone when you navigate away
      if (vadRef.current) {
        vadRef.current.destroy();
      }
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [startVad]);


  // Determine status text based on state
  const getStatusText = () => {
      if (error) return `Error: ${error}`;
      switch (assistantStatus) {
          case 'idle': return 'Listening...';
          case 'processing_stt': return 'Transcribing...';
          case 'processing_llm': return 'Thinking...';
          case 'speaking': return 'Speaking...';
          default: return 'Initializing...';
      }
  };

  return (
    <div className="voice-assistant-container">
      <img src="/img/ProjectZephy023LogoRenewal.png" alt="Project Zephyrine Logo" className="assistant-page-logo" />

      <div className="assistant-header">
        <h1>Embedded Voice Powered Assistant</h1>
      </div>

      <div className="conversation-window">
        {/* Pass amplitude only when listening or speaking */}
        <VoiceVisualizer
          status={assistantStatus}
          amplitude={(assistantStatus === 'idle' || assistantStatus === 'speaking') ? amplitude : 0}
        />
        {/* We need a new element for status text if we remove it from the visualizer */}
        <div className="status-text-main">{getStatusText()}</div>
      </div>

      {/* The input area is now gone as it's fully automatic */}
    </div>
  );
};

export default VoiceAssistantPage;