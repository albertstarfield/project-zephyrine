// src/components/ShuttleDisplay.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import PrimaryFlightDisplay from './shuttle-displays/PrimaryFlightDisplay';
import FlightControlSurfaceDisplay from './shuttle-displays/FlightControlSurfaceDisplay';
import EngineSystemsDisplay from './shuttle-displays/EngineSystemsDisplay';
import '../styles/components/_shuttleDisplay.css';
import '../styles/components/_hsiDisplay.css'; // <-- ADD THIS IMPORT


const INSTRUMENT_DATA_API_URL = '/instrumentviewportdatastreamlowpriopreview';

const ShuttleDisplay = () => {
  const [instrumentData, setInstrumentData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [activeDisplay, setActiveDisplay] = useState('PFD'); 

  const navigate = useNavigate();

  useEffect(() => {
    // ... (data fetching logic remains the same)
    const eventSource = new EventSource(INSTRUMENT_DATA_API_URL);
    eventSource.onopen = () => setErrorMessage(null);
    eventSource.onmessage = (event) => {
      try { setInstrumentData(JSON.parse(event.data)); }
      catch (error) { setErrorMessage("Data parse error."); }
    };
    eventSource.onerror = () => {
      setErrorMessage("SSE connection error.");
      eventSource.close();
    };
    return () => eventSource.close();
  }, []);

  const data = instrumentData ? instrumentData.data : null;
  const systems = data ? data.systems : {};
  const flight_controls = systems ? systems.flight_controls : {};
  const status = systems ? systems.status : {};

  const handleGoToMainDisplay = () => {
    window.location.reload();
  };

  const renderActiveDisplay = () => {
    if (errorMessage) return <div className="error-message">{errorMessage}</div>;
    if (!data) return <div className="loading-message">Connecting to Instruments...</div>;
    
    switch (activeDisplay) {
      case 'PFD':
        return <PrimaryFlightDisplay data={data} />;
      case 'FCS':
        return <div className="fullscreen-panel-wrapper"><FlightControlSurfaceDisplay controls={flight_controls} status={status} /></div>;
      case 'OMS':
        return <div className="fullscreen-panel-wrapper"><EngineSystemsDisplay systems={systems} /></div>;
      default:
        return <PrimaryFlightDisplay data={data} />;
    }
  };

  return (
    <div className="shuttle-display-container">
      <div className="display-screen">
        {renderActiveDisplay()}
      </div>
      
      <div className="display-menu">
        <button id="zephoai-back-btn" onClick={handleGoToMainDisplay}>[zephOAI]</button>
        <button onClick={() => setActiveDisplay('PFD')} className={activeDisplay === 'PFD' ? 'active' : ''}>A/E PFD</button>
        <button onClick={() => setActiveDisplay('OMS')} className={activeDisplay === 'OMS' ? 'active' : ''}>OMS/MPS</button>
        <button onClick={() => setActiveDisplay('FCS')} className={activeDisplay === 'FCS' ? 'active' : ''}>FCS</button>
        <div className="menu-spacer"></div>
        <button>MEDS MSG RST</button>
        <button>MEDS MSG ACK</button>
      </div>
    </div>
  );
};

export default ShuttleDisplay;