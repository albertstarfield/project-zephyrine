// src/components/PreSplashScreen.jsx
import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import '../styles/components/_preSplashScreen.css'; // We'll create this CSS file

// API endpoint for the primed status check
// Assuming the backend is running on localhost:3001 as configured in vite.config.js proxy
const PRIMED_READY_API_URL = '/primedready';

const PreSplashScreen = ({ onPrimedAndReady }) => {
  const [statusMessage, setStatusMessage] = useState("Initializing system...");
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    let intervalId;

    const checkPrimedStatus = async () => {
      try {
        const response = await fetch(PRIMED_READY_API_URL);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        setStatusMessage(data.status); // Update displayed status
        setIsReady(data.primed_and_ready); // Update ready state

        if (data.primed_and_ready) {
          clearInterval(intervalId); // Stop polling once ready
          onPrimedAndReady(); // Notify parent to proceed to next splash screen
        }
      } catch (error) {
        console.error("Pre-splash API check failed:", error);
        setStatusMessage(`System check failed: ${error.message}. Retrying...`);
        // Optionally, implement a backoff strategy here instead of constant 1s retry
      }
    };

    // Initial check
    checkPrimedStatus();

    // Set up polling every second
    intervalId = setInterval(checkPrimedStatus, 1000); // Poll every 1 second

    // Cleanup function
    return () => {
      clearInterval(intervalId); // Clear interval on component unmount
    };
  }, [onPrimedAndReady]); // Dependency array ensures effect re-runs if onPrimedAndReady changes (it's a useCallback, so it should be stable)

  return (
    <div className="pre-splash-screen-overlay">
      <div className="pre-splash-screen-content">
        <pre className="pre-splash-status-text">{statusMessage}</pre> {/* Use <pre> for mono font */}
      </div>
    </div>
  );
};

PreSplashScreen.propTypes = {
  onPrimedAndReady: PropTypes.func.isRequired, // Callback when ready
};

export default PreSplashScreen;