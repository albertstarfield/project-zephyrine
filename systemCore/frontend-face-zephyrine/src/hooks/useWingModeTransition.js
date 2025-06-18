// src/hooks/useWingModeTransition.js
import { useState, useEffect, useRef } from 'react';

// Define the cheat code sequence
const CHEAT_CODE = "aether278unveilyourwings";

export const useWingModeTransition = () => {
  // State to track if wing mode is active
  const [isWingModeActive, setIsWingModeActive] = useState(false);
  // Ref to store the buffer of typed characters
  const keyBufferRef = useRef('');

  useEffect(() => {
    const handleKeyDown = (event) => {
      // Append the pressed key to the buffer
      keyBufferRef.current += event.key.toLowerCase();

      // Keep the buffer length equal to the cheat code length
      if (keyBufferRef.current.length > CHEAT_CODE.length) {
        keyBufferRef.current = keyBufferRef.current.slice(-CHEAT_CODE.length);
      }

      // Check if the current buffer matches the cheat code
      if (keyBufferRef.current === CHEAT_CODE) {
        setIsWingModeActive(true);
        console.log("Cheat code 'aether278unveilyourwings' activated! Entering Zephy Avian Mode.");
        // Optionally, clear the buffer or reset it to prevent re-triggering immediately
        keyBufferRef.current = '';
      }
    };

    // Add event listener to the window for global keypress detection
    window.addEventListener('keydown', handleKeyDown);

    // Cleanup function: remove event listener when component unmounts
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []); // Empty dependency array means this effect runs once on mount

  return isWingModeActive;
};