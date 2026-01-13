// src/components/PreSplashScreen.jsx
import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import '../styles/components/_preSplashScreen.css';

const PRIMED_READY_API_URL = '/primedready';

const PreSplashScreen = ({ onPrimedAndReady }) => {
  const [statusMessage, setStatusMessage] = useState("Initializing system...");
  const [isReady, setIsReady] = useState(false);
  // NEW: State to track if the API is considered offline/unavailable
  const [isOffline, setIsOffline] = useState(false); 
  // NEW: Ref to track consecutive fetch failures
  const retryCountRef = useRef(0);
  // NEW: Threshold for consecutive failures before declaring offline
  const MAX_OFFLINE_RETRIES = 5; 

  useEffect(() => {
    let intervalId;

    const checkPrimedStatus = async () => {
      try {
        // Reset offline status and retry count on a successful attempt start
        if (retryCountRef.current === 0 && isOffline) { // Only reset if we're coming back from an offline state
             setIsOffline(false);
        }

        const response = await fetch(PRIMED_READY_API_URL);
        if (!response.ok) {
          retryCountRef.current++; // Increment retry count on HTTP error
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        // Reset retry count and offline status on successful fetch
        retryCountRef.current = 0; 
        setIsOffline(false); 
        
        setStatusMessage(data.status); // Update displayed status
        setIsReady(data.primed_and_ready); // Update ready state

        if (data.primed_and_ready) {
          clearInterval(intervalId); // Stop polling once ready
          onPrimedAndReady(); // Notify parent to proceed
        }
      } catch (error) {
        console.error("PreSplashScreen API check failed:", error);
        
        // If consecutive retries exceed threshold, set offline status
        if (retryCountRef.current >= MAX_OFFLINE_RETRIES) {
          setIsOffline(true);
          setStatusMessage("Guru Meditation : Zephy Cortex API Not Available");
        } else {
          // Keep showing retry message if below threshold
          setStatusMessage(`System check failed: ${error.message}. Retrying...`);
        }
      }
    };

    // Initial check immediately
    checkPrimedStatus();

    // Set up polling every second
    intervalId = setInterval(checkPrimedStatus, 1000); // Poll every 1 second

    // Cleanup function
    return () => {
      clearInterval(intervalId); // Clear interval on component unmount
    };
  }, [onPrimedAndReady, isOffline]); // Add isOffline to dependencies to trigger re-evaluation when offline state changes

  return (
    <div className="pre-splash-screen-overlay">
      <div className="pre-splash-screen-content">
        {/* Conditionally apply 'offline-error' class based on isOffline state */}
        <pre className={`pre-splash-status-text ${isOffline ? 'offline-error' : ''}`}>{statusMessage}</pre>
      </div>
    </div>
  );
};

PreSplashScreen.propTypes = {
  onPrimedAndReady: PropTypes.func.isRequired,
};

export default PreSplashScreen;