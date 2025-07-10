// src/components/shuttle-displays/PrimaryFlightDisplay.jsx

import React from 'react';
import VerticalTape from './VerticalTape';
import AttitudeIndicator from './AttitudeIndicator';
import GForceMeter from './GForceMeter';
//import PfdHsi from './PfdHsi';
import HorizontalSituationIndicatorPage from './HorizontalSituationIndicatorPage';

// This component is now ONLY the PFD view.
const PrimaryFlightDisplay = ({ data }) => {
  const { flight_dynamics = {}, navigation = {}, systems = {} } = data;
  const { attitude = {}, rates = {}, velocity = {}, position = {}, aero = {} } = flight_dynamics;
  const { flight_director = {}, hsi = {} } = navigation;
  const { status = {} } = systems;
  
  const formatValue = (value, decimals = 0) => (typeof value === 'number' ? value.toFixed(decimals) : 'N/A');

  return (
    <div className="pfd-grid">
      {/* --- Top Row --- */}
      <div className="pfd-dap-status">
        DAP: {status.dap_mode || 'N/A'}<br />
        Throt: {status.throttle_mode || 'N/A'}
      </div>
      <div className="pfd-rate-status">
        R {formatValue(rates.roll_rate, 0).padStart(3, '0')}<br />
        P {formatValue(rates.pitch_rate, 0).padStart(3, '0')}<br />
        Y {formatValue(rates.yaw_rate, 0).padStart(3, '0')}
      </div>
      <div className="pfd-mm-status">
        MM: 103<br/>
        ATT: {status.attitude_ref || 'N/A'}
      </div>
      
      {/* --- Middle Row --- */}
      <div className="pfd-speed-tapes">
        <VerticalTape
          value={velocity.mach}
          label="M/VI"
          range={[-0.5, 10]}
          boxValueFormatter={(v) => formatValue(v, 2)}
        />
        <VerticalTape
          value={aero.angle_of_attack}
          label="α"
          range={[-20, 20]}
          step={5}
          boxValueFormatter={(v) => formatValue(v, 1)}
          labelFormatter={(v) => v.toFixed(0)}
        />
      </div>
      <div className="pfd-adi-container">
        <AttitudeIndicator
          roll={attitude.roll || 0}
          pitch={attitude.pitch || 0}
          cmdRoll={flight_director.command_roll || 0}
          cmdPitch={flight_director.command_pitch || 0}
          yaw_rate={rates.yaw_rate || 0}
          sideslip_angle={aero.sideslip_angle || 0}
        />
      </div>
      <div className="pfd-altitude-tapes">
        <VerticalTape
          value={position.altitude}
          label="H"
          range={[ (position.altitude) - 10000, (position.altitude) + 10000 ]}
          step={5000}
          boxValueFormatter={(v) => `${formatValue(v / 1000, 0)}K`}
          labelFormatter={(v) => `${v / 1000}K`}
        />
        <VerticalTape
          value={velocity.vertical_speed}
          label="Ḣ"
          range={[ (velocity.vertical_speed) - 2000, (velocity.vertical_speed) + 2000 ]}
          step={500}
          boxValueFormatter={(v) => formatValue(v, 0)}
          labelFormatter={(v) => `${(v / 1000).toFixed(1)}K`}
        />
      </div>

      {/* --- Bottom Row --- */}
      <div className="pfd-airspeed-readouts">
         <div className="readout-box">{formatValue(velocity.keas, 0)}</div>KEAS
         <div className="readout-box">R {formatValue(aero.sideslip_angle, 1)}</div>Beta
      </div>
      <div className="pfd-center-bottom">
        <div className="pfd-g-meter-wrapper"><GForceMeter g={aero.g_force || 0} /></div>
        <div className="pfd-hsi-wrapper full-hsi">
            <HorizontalSituationIndicatorPage data={data} />
        </div>
      </div>
      <div className="pfd-nav-readouts">
         <div className="readout-box">{hsi.x_track_error?.toFixed(1) || 'N/A'}</div>X-Trk
         <div className="readout-box">{hsi.delta_inc?.toFixed(2) || 'N/A'}</div>Δ Inc
      </div>
    </div>
  );
};

export default PrimaryFlightDisplay;