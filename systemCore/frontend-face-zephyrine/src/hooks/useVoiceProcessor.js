import { useState, useEffect, useRef, useCallback } from 'react';
// The library exports MicVAD, not a default VAD object.
import { MicVAD } from '@ricky0123/vad-web';

export function useVoiceProcessor() {
  const [voiceState, setVoiceState] = useState('idle'); // idle, listening, processing, speaking
  const [error, setError] = useState(null);
  const [transcribedText, setTranscribedText] = useState({ text: '', final: false });
  
  const vadRef = useRef(null);
  const audioChunksRef = useRef([]);

  // This function will be called with the final audio blob when speech ends
  const processFinalAudio = useCallback(async (audioBlob) => {
    console.log("Hook: Final audio blob received for processing.", audioBlob);
    // In the next and final step, this is where we will put the 3 fetch calls.
    // For now, it just signals that processing has started.
    setVoiceState('processing');
    
    // --- SIMULATION FOR NOW ---
    // This will be replaced with real API calls
    await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time
    console.log("Hook: Simulation complete. Setting transcribed text.");
    setTranscribedText({ text: "This is the final transcription from our simulated API call.", final: true });
    setVoiceState('speaking'); // Simulate moving to speaking state
    await new Promise(resolve => setTimeout(resolve, 3000)); // Simulate speaking time
    setVoiceState('idle'); // Reset to idle
    // --- END SIMULATION ---
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setVoiceState('idle');
    try {
      console.log("VoiceProcessor: Initializing MicVAD...");
      // We use MicVAD.new() which handles microphone access and VAD processing together
      const myVad = await MicVAD.new({
        onSpeechEnd: (audio) => {
          // onSpeechEnd provides the audio as a Float32Array.
          // We need to convert it to a Blob to send to an API.
          console.log("VAD: Speech ended. Creating Blob.");
          const wavBuffer = new ArrayBuffer(44 + audio.length * 2);
          const view = new DataView(wavBuffer);
          // Standard WAV header
          writeUTFBytes(view, 0, 'RIFF');
          view.setUint32(4, 36 + audio.length * 2, true);
          writeUTFBytes(view, 8, 'WAVE');
          writeUTFBytes(view, 12, 'fmt ');
          view.setUint32(16, 16, true);
          view.setUint16(20, 1, true);
          view.setUint16(22, 1, true);
          view.setUint32(24, 16000, true); // Sample rate
          view.setUint32(28, 16000 * 2, true);
          view.setUint16(32, 2, true);
          view.setUint16(34, 16, true);
          writeUTFBytes(view, 36, 'data');
          view.setUint32(40, audio.length * 2, true);
          // Write PCM data
          let offset = 44;
          for (let i = 0; i < audio.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, audio[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
          }
          const audioBlob = new Blob([view], { type: 'audio/wav' });
          processFinalAudio(audioBlob);
        },
        onSpeechStart: () => {
          console.log("VAD: Speech started.");
          setVoiceState('listening');
        },
        // You can still add other VAD options here
      });
      myVad.start();
      vadRef.current = myVad;
      
    } catch (err) {
      console.error("VoiceProcessor: MicVAD Initialization Error:", err);
      setError("Microphone access was denied or an error occurred. Please check your browser permissions.");
    }
    
    // Helper function for WAV header
    function writeUTFBytes(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

  }, [processFinalAudio]);

  const stop = useCallback(() => {
    if (vadRef.current) {
      console.log("VoiceProcessor: Destroying VAD and releasing microphone.");
      vadRef.current.destroy();
      vadRef.current = null;
    }
    setVoiceState('idle');
  }, []);

  return { start, stop, voiceState, error, transcribedText };
}