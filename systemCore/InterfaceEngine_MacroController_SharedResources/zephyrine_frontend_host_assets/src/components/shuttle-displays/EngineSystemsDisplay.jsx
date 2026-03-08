// src/components/shuttle-displays/EngineSystemsDisplay.jsx
import React from 'react';

// Reusable sub-component for a single pressure gauge
const PressureGauge = ({ title, value = 0, max = 3000, units, label }) => {
  const heightPct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="pressure-gauge">
      <div className="gauge-label top">{title}</div>
      <div className="gauge-bar">
        <div className="gauge-level" style={{ height: `${heightPct}%` }}></div>
      </div>
      <div className="gauge-label bottom">{units}</div>
      <div className="gauge-label value">{value.toFixed(0)}</div>
    </div>
  );
};

const EngineSystemsDisplay = ({ systems = {} }) => {
  // Gracefully access nested data from your JSON
  const { oms = {}, mps = {}, pneu = {} } = systems;
  const { eng_manf = {} } = mps;

  return (
    <div className="engine-panel-container">
      {/* --- OMS Column --- */}
      <div className="engine-column">
        <div className="column-title">OMS</div>
        <div className="gauge-row">
          <PressureGauge title="L" units="He TK P" value={oms.he_tk_press_l} max={4000} />
          <PressureGauge title="R" units="He TK P" value={oms.he_tk_press_r} max={4000} />
        </div>
        <div className="gauge-row">
          <PressureGauge title="L" units="N2 TK P" value={oms.n2_tk_press_l} max={4000} />
          <PressureGauge title="R" units="N2 TK P" value={oms.n2_tk_press_r} max={4000} />
        </div>
        <div className="gauge-row">
          <PressureGauge title="L" units="Pc %" value={oms.pc_l} max={100} />
          <PressureGauge title="R" units="Pc %" value={oms.pc_r} max={100} />
        </div>
      </div>

      {/* --- Center Column (PNEU / MANIFOLD) --- */}
      <div className="engine-column center">
        <div className="gauge-row single">
          <PressureGauge title="PNEU" units="He TK P" value={pneu.he_tk_press} max={4000} />
        </div>
        <div className="gauge-row single">
          <PressureGauge title="REG" units="P" value={pneu.he_reg_press} max={4000} />
        </div>
        <div className="gauge-row single">
           <PressureGauge title="ENG MANF" units="LO2" value={eng_manf.lo2_press} max={4000} />
        </div>
         <div className="gauge-row single">
           <PressureGauge title="" units="LH2" value={eng_manf.lh2_press} max={4000} />
        </div>
      </div>

      {/* --- MPS Column --- */}
      <div className="engine-column">
        <div className="column-title">MPS</div>
        <div className="gauge-row">
          <PressureGauge title="L/2" units="He TK P" value={mps.he_tk_press_l} max={4000} />
          <PressureGauge title="C/1" units="He TK P" value={mps.he_tk_press_c} max={4000} />
          <PressureGauge title="R/3" units="He TK P" value={mps.he_tk_press_r} max={4000} />
        </div>
        <div className="gauge-row">
           <PressureGauge title="L/2" units="He REG P" value={mps.he_reg_press_l} max={4000} />
           <PressureGauge title="C/1" units="He REG P" value={mps.he_reg_press_c} max={4000} />
           <PressureGauge title="R/3" units="He REG P" value={mps.he_reg_press_r} max={4000} />
        </div>
        <div className="gauge-row">
          <PressureGauge title="L/2" units="Pc %" value={mps.pc_l} max={100} />
          <PressureGauge title="C/1" units="Pc %" value={mps.pc_c} max={100} />
          <PressureGauge title="R/3" units="Pc %" value={mps.pc_r} max={100} />
        </div>
      </div>
    </div>
  );
};

export default EngineSystemsDisplay;