// src/components/shuttle-displays/VerticalTape.jsx
import React from 'react';

const VerticalTape = ({ value, label, range, boxValue }) => {
  const [min, max] = range;
  const percentage = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));

  // Create tick marks
  const ticks = [];
  for (let i = min; i <= max; i += (max - min) / 4) {
    ticks.push(i);
  }

  return (
    <div className="vertical-tape">
      <div className="tape-label">{label}</div>
      <div className="tape-body">
        <div className="tape-indicator" />
        <div className="tape-scale" style={{ bottom: `${percentage}%` }}>
          {ticks.map(tick => (
            <div key={tick} className="tape-tick">
              <span>{Math.round(tick)}</span>
              <div className="tape-line" />
            </div>
          ))}
        </div>
      </div>
      <div className="tape-box">{boxValue}</div>
    </div>
  );
};

export default VerticalTape;