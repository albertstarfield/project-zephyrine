// src/components/ShuttleDisplay.jsx
import React, { useState, useEffect } from 'react';
import PrimaryFlightDisplay from './shuttle-displays/PrimaryFlightDisplay';
import HorizontalSituationIndicatorPage from './shuttle-displays/HorizontalSituationIndicatorPage.jsx';
import FlightControlsPage from './shuttle-displays/FlightControlsPage';
import '../styles/components/_shuttleDisplay.css'; // We will create this CSS file

const INSTRUMENT_DATA_API_URL = '/api/instrumentviewportdatastreamlowpriopreview';

const ShuttleDisplay = () => {
  const [instrumentData, setInstrumentData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [activeDisplay, setActiveDisplay] = useState('PFD');

  useEffect(() => {
    // This data fetching logic is the same as your WingModePage
    const eventSource = new EventSource(INSTRUMENT_DATA_API_URL);
    eventSource.onopen = () => setErrorMessage(null);
    eventSource.onmessage = (event) => {
      try {
        setInstrumentData(JSON.parse(event.data));
      } catch (error) {
        console.error("Error parsing SSE data:", error);
        setErrorMessage("Data parse error.");
      }
    };
    eventSource.onerror = () => {
      setErrorMessage("SSE connection error.");
      eventSource.close();
    };
    return () => eventSource.close();
  }, []);

  const data = instrumentData ? instrumentData.data : null;

  const renderActiveDisplay = () => {
    if (errorMessage) return <div className="error-message">{errorMessage}</div>;
    if (!data) return <div className="loading-message">Connecting to Instruments...</div>;
    
    switch (activeDisplay) {
      case 'PFD':
        return <PrimaryFlightDisplay data={data} />;
      case 'HSI':
        return <HorizontalSituationIndicatorPage data={data} />;
      case 'FCS':
        return <FlightControlsPage data={data} />;
      default:
        return <PrimaryFlightDisplay data={data} />;
    }
  };

  return (
    <div className="shuttle-display-container">
      <div className="display-screen">
        {renderActiveDisplay()}
      </div>
      {/* This menu simulates the bottom buttons */}
      <div className="display-menu">
        <button onClick={() => setActiveDisplay('PFD')} className={activeDisplay === 'PFD' ? 'active' : ''}>A/E PFD</button>
        <button onClick={() => setActiveDisplay('HSI')} className={activeDisplay === 'HSI' ? 'active' : ''}>ORBIT PFD</button>
        <button onClick={() => setActiveDisplay('FCS')} className={activeDisplay === 'FCS' ? 'active' : ''}>FCS</button>
        <div className="menu-spacer"></div>
        <button>MEDS MSG RST</button>
        <button>MEDS MSG ACK</button>
      </div>
    </div>
  );
};

export default ShuttleDisplay;