// src/components/WingModePage.jsx
import React, { useState, useEffect } from 'react';
import '../styles/components/_wingModePage.css';

// API endpoint for the instrument data stream (now proxied via backend)
const INSTRUMENT_DATA_API_URL = '/instrumentviewportdatastreamlowpriopreview';

const WingModePage = () => {
  const [instrumentData, setInstrumentData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  useEffect(() => {
    const eventSource = new EventSource(INSTRUMENT_DATA_API_URL);

    eventSource.onopen = () => {
      console.log("SSE Connection opened for instrument data.");
      setErrorMessage(null);
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setInstrumentData(data);
        setErrorMessage(null);
      } catch (error) {
        console.error("Error parsing SSE message data:", error, event.data);
        setErrorMessage("Data parse error. Check console for raw data.");
      }
    };

    eventSource.onerror = (error) => {
      console.error("SSE Error:", error);
      if (error.status) {
          setErrorMessage(`SSE connection error: Status ${error.status}`);
      } else if (error.message) {
          setErrorMessage(`SSE connection error: ${error.message}`);
      } else {
          setErrorMessage("SSE connection error. See console for details.");
      }
      eventSource.close();
    };

    return () => {
      console.log("Closing SSE connection for instrument data.");
      eventSource.close();
    };
  }, []);

  const renderTileContent = (tileNum) => {
    if (errorMessage) {
      return <span style={{ color: 'red' }}>{errorMessage}</span>;
    }
    if (!instrumentData) {
      return "Connecting to instruments...";
    }

    // Safely access the 'data' sub-object and its nested properties
    const data = instrumentData.data || {};
    const {
      flight_dynamics = {},
      navigation = {},
      systems = {},
      timestamp // Root level timestamp in 'data'
    } = data;

    // Destructure flight_dynamics
    const {
      attitude = {},
      rates = {},
      velocity = {},
      position = {},
      aero = {}
    } = flight_dynamics;

    // Destructure navigation
    const {
      flight_director = {},
      hsi = {}
    } = navigation;

    // Destructure systems
    const {
      status = {},
      flight_controls = {}
    } = systems;

    // Helper to format numbers or return N/A
    const formatNum = (value, unit = '') => value !== undefined && value !== null ? `${value.toFixed(2)}${unit}` : "N/A";
    const formatDeg = (value) => formatNum(value, '°');
    const formatPct = (value) => formatNum(value, '%');


    switch (tileNum) {
      case 1: // General Flight Status & Position
        return (
          <>
            <h3>Mode</h3>
            <p>{flight_dynamics.mode || "N/A"}</p> {/* Assuming mode is directly under flight_dynamics or root */}
            <h3>Altitude</h3>
            <p>{formatNum(position.altitude, 'm')}</p>
            <h3>Vertical Speed</h3>
            <p>{formatNum(velocity.vertical_speed, ' m/s')}</p>
            <h3>Source Daemon</h3>
            <p>{instrumentData.source_daemon || "N/A"}</p>
            <h3>Timestamp</h3>
            <p>{new Date(timestamp || instrumentData.timestamp_py).toLocaleTimeString()}</p>
          </>
        );
      case 2: // Attitude & Rates
        return (
          <>
            <h3>Attitude (P/R/Y)</h3>
            <p>P: {formatDeg(attitude.pitch)}</p>
            <p>R: {formatDeg(attitude.roll)}</p>
            <p>Y: {formatDeg(attitude.yaw)}</p>
            <h3>Rates (P/R/Y)</h3>
            <p>P: {formatNum(rates.pitch_rate, '°/s')}</p>
            <p>R: {formatNum(rates.roll_rate, '°/s')}</p>
            <p>Y: {formatNum(rates.yaw_rate, '°/s')}</p>
          </>
        );
      case 3: // Velocity & Aerodynamics
        return (
          <>
            <h3>Velocity</h3>
            <p>Mach: {formatNum(velocity.mach)}</p>
            <p>KEAS: {formatNum(velocity.keas, 'kts')}</p>
            <h3>Aerodynamics</h3>
            <p>G-Force: {formatNum(aero.g_force, 'g')}</p>
            <p>AoA: {formatDeg(aero.angle_of_attack)}</p>
            <p>Sideslip: {formatDeg(aero.sideslip_angle)}</p>
          </>
        );
      case 4: // Navigation & Systems Status
        const waypoints = hsi.waypoints || [];
        const firstWaypoint = waypoints[0];

        return (
          <>
            <h3>Flight Director</h3>
            <p>Cmd P: {formatDeg(flight_director.command_pitch)}</p>
            <p>Cmd R: {formatDeg(flight_director.command_roll)}</p>
            <h3>HSI</h3>
            <p>Course: {formatDeg(hsi.selected_course)}</p>
            <p>Deviation: {formatNum(hsi.course_deviation)}</p>
            {firstWaypoint && (
              <p>WP: {firstWaypoint.label || firstWaypoint.id} ({formatDeg(firstWaypoint.bearing)}, {formatNum(firstWaypoint.range, 'km')})</p>
            )}
            <h3>Systems Status</h3>
            <p>DAP: {status.dap_mode || "N/A"} | THR: {status.throttle_mode || "N/A"}</p>
            <p>FC: {status.flight_controller_mode || "N/A"} | YD: {status.yaw_damper_on ? "ON" : "OFF"}</p>
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