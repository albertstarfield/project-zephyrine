// src/components/shuttle-displays/FlightControlSurfaceDisplay.jsx
import React from 'react';

// Helper function to calculate a value's percentage on a scale
const calculatePercentage = (value, min, max) => {
  const clampedValue = Math.max(min, Math.min(max, value));
  return ((clampedValue - min) / (max - min)) * 100;
};

const FlightControlSurfaceDisplay = ({ controls, status }) => {
  // Gracefully handle missing data
  const elevonL = controls?.elevons?.left_deg || 0;
  const elevonR = controls?.elevons?.right_deg || 0;
  const bodyFlap = controls?.body_flap?.actual_pct || 0;
  const rudder = controls?.rudder?.actual_deg || 0;
  const aileron = (elevonL - elevonR) / 2; // Aileron is differential elevon
  const speedbrakeCmd = controls?.speedbrake?.commanded_pct || 0;
  const speedbrakeAct = controls?.speedbrake?.actual_pct || 0;
  
  return (
    <div className="fcs-container">
      {/* --- DAP/Thrust Status Header --- */}
      <div className="fcs-header">
        <div>DAP: {status?.dap_mode || 'N/A'}</div>
        <div>Throt: {status?.throttle_mode || 'N/A'}</div>
      </div>

      <div className="fcs-main-panel">
        {/* --- Vertical Gauges --- */}
        <div className="fcs-vertical-gauges">
          <div className="fcs-gauge-group">
            <div className="fcs-label">ELEVONS<br/>DEG</div>
            <div className="fcs-v-gauge">
              <div className="fcs-v-track">
                <div className="fcs-pointer diamond" style={{ top: `${100 - calculatePercentage(elevonL, -30, 20)}%` }}></div>
              </div>
              <div className="fcs-v-track">
                <div className="fcs-pointer diamond" style={{ top: `${100 - calculatePercentage(elevonR, -30, 20)}%` }}></div>
              </div>
            </div>
            <div className="fcs-v-scale">
              <span>-30</span><span>-20</span><span>-10</span><span>0</span><span>+10</span><span>+20</span>
            </div>
          </div>
          <div className="fcs-gauge-group">
            <div className="fcs-label">BODY FLAP<br/>%</div>
            <div className="fcs-v-gauge single">
              <div className="fcs-v-track">
                <div className="fcs-pointer triangle-right" style={{ top: `${100 - calculatePercentage(bodyFlap, 0, 100)}%` }}></div>
              </div>
            </div>
             <div className="fcs-v-scale flap">
              <span>0</span><span>20</span><span>40</span><span>60</span><span>80</span><span>100</span>
            </div>
          </div>
        </div>

        {/* --- Horizontal Gauges --- */}
        <div className="fcs-horizontal-gauges">
          <div className="fcs-h-gauge-wrapper">
             <div className="fcs-label">RUDDER-DEG</div>
             <div className="fcs-h-track">
                <div className="fcs-pointer triangle-up" style={{ left: `${calculatePercentage(rudder, -30, 30)}%` }}></div>
             </div>
             <div className="fcs-h-scale"><span>30</span><span>20</span><span>10</span><span>0</span><span>10</span><span>20</span><span>30</span></div>
             <div className="fcs-h-sublabel"><span>L RUD</span><span>R RUD</span></div>
          </div>
          <div className="fcs-h-gauge-wrapper">
             <div className="fcs-label">AILERON-DEG</div>
             <div className="fcs-h-track">
                <div className="fcs-pointer triangle-up" style={{ left: `${calculatePercentage(aileron, -5, 5)}%` }}></div>
             </div>
             <div className="fcs-h-scale"><span>5</span><span></span><span></span><span>0</span><span></span><span></span><span>5</span></div>
             <div className="fcs-h-sublabel"><span>L AIL</span><span>R AIL</span></div>
          </div>
           <div className="fcs-h-gauge-wrapper speedbrake">
             <div className="fcs-label">SPEEDBRAKE %</div>
             <div className="fcs-h-track">
                <div className="fcs-pointer triangle-up cmd" style={{ left: `${calculatePercentage(speedbrakeCmd, 0, 100)}%` }}></div>
                <div className="fcs-pointer diamond blue" style={{ left: `${calculatePercentage(speedbrakeAct, 0, 100)}%` }}></div>
             </div>
             <div className="fcs-h-scale"><span>0</span><span>20</span><span>40</span><span>60</span><span>80</span><span>100</span></div>
             <div className="fcs-h-sublabel-sb">
                <div className="fcs-sb-box">{String(speedbrakeAct).padStart(3,'0')}</div>
                <span>ACTUAL</span>
                <span>COMMAND</span>
                <div className="fcs-sb-box">{String(speedbrakeCmd).padStart(3,'0')}</div>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FlightControlSurfaceDisplay;