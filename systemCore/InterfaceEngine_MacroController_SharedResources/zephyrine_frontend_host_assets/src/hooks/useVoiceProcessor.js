// externalAnalyzer/frontend-face-zephyrine/src/hooks/useVoiceProcessor.js
import { useState, useEffect, useRef, useCallback } from 'react';
import { MicVAD } from '@ricky0123/vad-web';

// Centralize API configuration based on user feedback
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:11434/v1';
const AUDIO_TRANSCRIBE_API_URL = `${API_BASE_URL}/audio/transcriptions`;
const CHAT_COMPLETION_API_URL = `${API_BASE_URL}/chat/completions`;
const AUDIO_SPEECH_API_URL = `${API_BASE_URL}/audio/speech`;

const LLM_API_KEY = import.meta.env.VITE_OPENAI_API_KEY || 'ollama'; // Your LLM API key

/**
 * A custom hook to process voice input from the user, send it to an AI, and play back the response.
 * @param {boolean} shouldBeActive - An external signal to control whether the mic should be listening.
 * @param {object} user - The user object, used to gate activation.
 * @returns {object} The state of the voice processor and functions to control it.
 */
export function useVoiceProcessor(shouldBeActive = false, user = null) {
  // A state machine to manage the voice processing flow
  // idle | initializing | listening | transcribing | processing | speaking
  const [voiceState, setVoiceState] = useState('idle');
  const [error, setError] = useState(null);
  const [transcribedText, setTranscribedText] = useState('');
  
  const vadRef = useRef(null); // Reference to the MicVAD instance
  const audioContextRef = useRef(null); // Reference to the AudioContext for playback

  // Ref to hold the current state. This is crucial for breaking the useEffect dependency cycle.
  const voiceStateRef = useRef(voiceState);
  useEffect(() => {
    voiceStateRef.current = voiceState;
  }, [voiceState]);

  // --- Core Functions ---

  const playAudio = useCallback(async (audioDataBlob) => {
    if (!audioDataBlob) {
      console.warn("No audio data to play.");
      setVoiceState('idle');
      return;
    }
    
    setVoiceState('speaking');
    try {
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }

      const arrayBuffer = await audioDataBlob.arrayBuffer();
      const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
      const source = audioContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);
      
      await new Promise(resolve => {
        source.onended = resolve;
        source.start(0);
      });
      console.log("Audio playback finished.");

    } catch (err) {
      console.error("Error playing audio:", err);
      setError("Failed to play audio response.");
    } finally {
      setVoiceState('idle'); 
    }
  }, []);

  const processFinalAudio = useCallback(async (audioBlob) => {
    console.log("Hook: Final audio blob received for processing.", audioBlob);
    setError(null);

    try {
      setVoiceState('transcribing');
      const transcribeFormData = new FormData();
      transcribeFormData.append('file', audioBlob, 'audio.wav');
      transcribeFormData.append('model', 'whisper-1');

      const transcribeResponse = await fetch(AUDIO_TRANSCRIBE_API_URL, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${LLM_API_KEY}` },
        body: transcribeFormData,
      });
      if (!transcribeResponse.ok) throw new Error(`Transcription failed: ${transcribeResponse.statusText}`);
      
      const transcribeResult = await transcribeResponse.json();
      const userText = transcribeResult.text;
      console.log("Hook: Transcribed Text:", userText);
      setTranscribedText(userText);
      
      setVoiceState('processing');
      const chatResponse = await fetch(CHAT_COMPLETION_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${LLM_API_KEY}` },
        body: JSON.stringify({ model: "llama3", messages: [{ role: "user", content: userText }] }),
      });
      if (!chatResponse.ok) throw new Error(`Chat completion failed: ${chatResponse.statusText}`);

      const chatResult = await chatResponse.json();
      const assistantResponseText = chatResult.choices[0]?.message?.content;
      if (!assistantResponseText) throw new Error("No response from chat completion.");
      console.log("Hook: Assistant Response Text:", assistantResponseText);

      const speechResponse = await fetch(AUDIO_SPEECH_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${LLM_API_KEY}` },
        body: JSON.stringify({ model: "tts-1", input: assistantResponseText, voice: "alloy" }),
      });
      if (!speechResponse.ok) throw new Error(`Text-to-speech failed: ${speechResponse.statusText}`);
      
      const responseAudioBlob = await speechResponse.blob();
      await playAudio(responseAudioBlob);

    } catch (err) {
      console.error("VoiceProcessor: API processing error:", err);
      setError(`Voice processing failed: ${err.message}`);
      setVoiceState('idle');
    }
  }, [playAudio, LLM_API_KEY]);

  const stopVadAndMic = useCallback(() => {
    if (vadRef.current) {
      vadRef.current.destroy();
      vadRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    setVoiceState('idle');
    setTranscribedText('');
  }, []);
  
  const startVadAndMic = useCallback(async () => {
    // Read from the ref to check the current state without creating a dependency.
    if (voiceStateRef.current !== 'idle') return;

    setVoiceState('initializing');
    setError(null);
    setTranscribedText('');

    try {
      console.log("VoiceProcessor: Initializing MicVAD...");
      
      const writeUTFBytes = (view, offset, string) => {
          for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
      };

      const myVad = await MicVAD.new({
        onSpeechStart: () => {
          console.log("VAD: Speech started.");
          setVoiceState('listening');
        },
        onSpeechEnd: (audio) => {
          console.log("VAD: Speech ended. Creating Blob.");
          stopVadAndMic();

          const wavBuffer = new ArrayBuffer(44 + audio.length * 2);
          const view = new DataView(wavBuffer);
          writeUTFBytes(view, 0, 'RIFF');
          view.setUint32(4, 36 + audio.length * 2, true);
          writeUTFBytes(view, 8, 'WAVE');
          writeUTFBytes(view, 12, 'fmt ');
          view.setUint32(16, 16, true);
          view.setUint16(20, 1, true);
          view.setUint16(22, 1, true);
          view.setUint32(24, 16000, true);
          view.setUint32(28, 16000 * 2, true);
          view.setUint16(32, 2, true);
          view.setUint16(34, 16, true);
          writeUTFBytes(view, 36, 'data');
          view.setUint32(40, audio.length * 2, true);
          let offset = 44;
          for (let i = 0; i < audio.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, audio[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
          }
          const audioBlob = new Blob([view], { type: 'audio/wav' });
          
          processFinalAudio(audioBlob);
        },
        onError: (e) => {
            console.error("VAD Error:", e);
            setError("A VAD error occurred. Please check permissions.");
            stopVadAndMic();
        }
      });
      
      myVad.start();
      vadRef.current = myVad;
      
    } catch (err) {
      console.error("VoiceProcessor: MicVAD Initialization Error:", err);
      setError("Microphone access was denied. Please check browser permissions.");
      stopVadAndMic();
    }
  }, [processFinalAudio, stopVadAndMic]); // Note: voiceState is not a dependency here.

  // --- Main Control Effect ---
  useEffect(() => {
    if (shouldBeActive && user && voiceState === 'idle') {
      startVadAndMic();
    } else if (!shouldBeActive && !['idle', 'speaking'].includes(voiceState)) {
      stopVadAndMic();
    }
  }, [shouldBeActive, user, voiceState, startVadAndMic, stopVadAndMic]);
  
  // Cleanup on unmount
  useEffect(() => {
      return () => {
          if (vadRef.current) {
              vadRef.current.destroy();
          }
          if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close();
          }
      }
  }, []);

  return { 
    voiceState,
    isInitializing: voiceState === 'initializing',
    isRecording: voiceState === 'listening',
    isSpeaking: voiceState === 'speaking',   
    isProcessing: voiceState === 'transcribing' || voiceState === 'processing',
    transcript: transcribedText, 
    error,
    activateMic: startVadAndMic,
    deactivateMic: stopVadAndMic,
  };
}
