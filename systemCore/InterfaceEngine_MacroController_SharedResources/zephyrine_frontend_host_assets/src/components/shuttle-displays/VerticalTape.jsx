// src/components/shuttle-displays/VerticalTape.jsx

import React, { useMemo } from 'react';

const VerticalTape = ({
  value = 0,
  label,
  range = [-100, 100],
  step = 20,
  boxValueFormatter = (v) => v.toFixed(0),
  labelFormatter = (v) => v.toString(),
}) => {
  const [min, max] = range;

  // Calculate the position of the moving scale as a percentage
  const scalePosition = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));

  // Generate the ticks and labels for the scale
  const ticks = useMemo(() => {
    const generatedTicks = [];
    for (let i = min; i <= max; i += step) {
      generatedTicks.push({
        value: i,
        label: labelFormatter(i),
        position: ((i - min) / (max - min)) * 100,
      });
    }
    return generatedTicks;
  }, [min, max, step, labelFormatter]);

  return (
    <div className="vertical-tape">
      <div className="tape-label">{label}</div>
      <div className="tape-body">
        {/* The moving scale with ticks and labels */}
        <div className="tape-scale" style={{ bottom: `${scalePosition}%` }}>
          {ticks.map(tick => (
            <div key={tick.value} className="tape-tick" style={{ bottom: `${tick.position}%` }}>
              <span>{tick.label}</span>
              <div className="tape-line" />
            </div>
          ))}
        </div>

        {/* The fixed central indicator with the digital readout */}
        <div className="tape-indicator-wrapper">
          <div className="tape-indicator-box">
            {boxValueFormatter(value)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VerticalTape;