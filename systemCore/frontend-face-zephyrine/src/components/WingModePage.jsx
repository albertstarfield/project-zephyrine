// src/components/WingModePage.jsx
import React, { useState, useEffect } from 'react';
import '../styles/components/_wingModePage.css';

// API endpoint for the instrument data stream (now proxied via backend)
const INSTRUMENT_DATA_API_URL = '/api/instrumentviewportdatastreamlowpriopreview';

const WingModePage = () => {
  const [instrumentData, setInstrumentData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  useEffect(() => {
    // EventSource is designed for Server-Sent Events (SSE)
    const eventSource = new EventSource(INSTRUMENT_DATA_API_URL);

    eventSource.onopen = () => {
      console.log("SSE Connection opened for instrument data.");
      setErrorMessage(null); // Clear errors on successful open
    };

    eventSource.onmessage = (event) => {
      try {
        // Each event.data will be a JSON string from the "data:" line
        const data = JSON.parse(event.data);
        setInstrumentData(data); // Update state with the latest data object
        setErrorMessage(null); // Clear errors if data is successfully received
      } catch (error) {
        console.error("Error parsing SSE message data:", error, event.data);
        setErrorMessage("Data parse error. Check console.");
      }
    };

    eventSource.onerror = (error) => {
      console.error("SSE Error:", error);
      // Attempt to get more specific error message
      if (error.status) { // For Fetch/XHR errors
          setErrorMessage(`SSE connection error: Status ${error.status}`);
      } else if (error.message) { // For generic JS errors
          setErrorMessage(`SSE connection error: ${error.message}`);
      } else {
          setErrorMessage("SSE connection error. See console for details.");
      }
      eventSource.close(); // Close the connection on error to avoid infinite retries if server isn't responsive
    };

    // Cleanup: Close the EventSource connection when the component unmounts
    return () => {
      console.log("Closing SSE connection for instrument data.");
      eventSource.close();
    };
  }, []); // Empty dependency array means this effect runs once on mount and cleans up on unmount

  const renderTileContent = (tileNum) => {
    if (errorMessage) {
      return <span style={{ color: 'red' }}>{errorMessage}</span>;
    }
    if (!instrumentData) {
      return "Connecting to instruments...";
    }

    const data = instrumentData.data;
    // Safely destructure data, providing defaults to prevent errors if keys are missing
    const {
      mode = "N/A",
      altimeter,
      attitude_indicator,
      autopilot_status,
      airspeed_indicator,
      gps_speed,
      heading_indicator,
      turn_coordinator,
      vertical_speed_indicator,
      navigation_reference,
      relative_velocity_c
    } = data || {}; // Ensure data is not null before destructuring

    switch (tileNum) {
      case 1: // General Status / Mode
        return (
          <>
            <h3>Mode</h3>
            <p>{mode}</p>
            <h3>Source Daemon</h3>
            <p>{instrumentData.source_daemon || "N/A"}</p>
            <h3>Timestamp</h3>
            <p>{new Date(data?.timestamp || instrumentData.timestamp_py).toLocaleTimeString()}</p>
          </>
        );
      case 2: // Altitude / Vertical Speed
        return (
          <>
            <h3>Altimeter</h3>
            <p>{altimeter !== undefined ? `${altimeter.toFixed(2)}m` : "N/A"}</p>
            <h3>Vertical Speed</h3>
            <p>{vertical_speed_indicator !== undefined ? `${vertical_speed_indicator.toFixed(2)} m/s` : "N/A"}</p>
            {mode === "Interstellar Flight" && relative_velocity_c !== undefined && (
              <>
                <h3>Rel Velocity (c)</h3>
                <p>{relative_velocity_c.toFixed(4)}c</p>
              </>
            )}
          </>
        );
      case 3: // Speed / Heading / Navigation
        return (
          <>
            <h3>Airspeed</h3>
            <p>{airspeed_indicator !== undefined ? `${airspeed_indicator.toFixed(2)} km/h` : "N/A"}</p>
            <h3>GPS Speed</h3>
            <p>{gps_speed !== undefined ? `${gps_speed.toFixed(2)} km/h` : "N/A"}</p>
            <h3>Heading</h3>
            <p>{heading_indicator !== undefined ? `${heading_indicator.toFixed(1)}°` : "N/A"}</p>
            {mode === "Interstellar Flight" && navigation_reference && (
              <>
                <h3>Nav Ref</h3>
                <p>{navigation_reference}</p>
              </>
            )}
          </>
        );
      case 4: // Attitude / Autopilot / Turn
        const pitch = attitude_indicator?.pitch;
        const roll = attitude_indicator?.roll;
        const apStatus = autopilot_status;
        const turnRate = turn_coordinator?.rate;
        const slipSkid = turn_coordinator?.slip_skid;

        return (
          <>
            <h3>Attitude</h3>
            <p>Pitch: {pitch !== undefined ? `${pitch.toFixed(1)}°` : "N/A"}</p>
            <p>Roll: {roll !== undefined ? `${roll.toFixed(1)}°` : "N/A"}</p>
            <h3>Autopilot</h3>
            <p>AP: {apStatus?.AP ? "ON" : "OFF"} | HDG: {apStatus?.HDG ? "ON" : "OFF"} | NAV: {apStatus?.NAV ? "ON" : "OFF"}</p>
            {turnRate !== undefined && (
              <>
                <h3>Turn Rate</h3>
                <p>{turnRate.toFixed(2)}°/s</p>
                <h3>Slip/Skid</h3>
                <p>{slipSkid !== undefined ? slipSkid.toFixed(1) : "N/A"}</p>
              </>
            )}
          </>
        );
      default:
        return `Tile ${tileNum} Content`;
    }
  };

  return (
    <div className="wing-mode-overlay">
      <div className="wing-mode-content">
        <h1 className="wing-mode-title">Zephy Avian Mode</h1>
        <div className="wing-mode-tiles-container">
          <div className="wing-mode-tile">{renderTileContent(1)}</div>
          <div className="wing-mode-tile">{renderTileContent(2)}</div>
          <div className="wing-mode-tile">{renderTileContent(3)}</div>
          <div className="wing-mode-tile">{renderTileContent(4)}</div>
        </div>
      </div>
    </div>
  );
};

export default WingModePage;