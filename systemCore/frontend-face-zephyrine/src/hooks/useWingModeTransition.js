// src/hooks/useWingModeTransition.js
import { useState, useEffect, useRef } from 'react';

// Define the cheat code sequence
const CHEAT_CODE = "aether278unveilyourwings";
const WING_MODE_STORAGE_KEY = "wingModeOverride";

export const useWingModeTransition = () => {
  // State to track if wing mode is active, initialized from localStorage
  const [isWingModeActive, setIsWingModeActive] = useState(() => {
    try {
      const storedValue = window.localStorage.getItem(WING_MODE_STORAGE_KEY);
      return storedValue === 'true';
    } catch (error) {
      console.error("Could not read wing mode override from localStorage", error);
      return false;
    }
  });

  // Ref to store the buffer of typed characters
  const keyBufferRef = useRef('');

  useEffect(() => {
    // If already active from storage, no need for the keydown listener
    if (isWingModeActive) return;

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
  }, [isWingModeActive]); // Re-run effect if wing mode is deactivated

  return isWingModeActive;
};