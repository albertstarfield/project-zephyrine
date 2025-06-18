// src/components/shuttle-displays/GForceMeter.jsx
import React from 'react';

const GForceMeter = ({ g }) => {
  // Define the geometry of the meter
  const centerX = 50;
  const centerY = 50;
  const radius = 40;
  const needleLength = 35;
  const startAngle = -135; // Corresponds to -1g
  const endAngle = 45;     // Corresponds to 3g
  const totalAngle = endAngle - startAngle; // 180 degrees total sweep
  const gRange = 4; // from -1 to 3

  // Calculate the needle rotation in degrees
  // Map the g-force value to the angle range
  const gClamped = Math.max(-1, Math.min(3, g));
  const rotation = startAngle + ((gClamped - (-1)) / gRange) * totalAngle;

  // Create the tick marks for the scale
  const ticks = [-1, 0, 1, 2, 3].map(gValue => {
    const angle = startAngle + ((gValue - (-1)) / gRange) * totalAngle;
    const rad = angle * (Math.PI / 180);
    const x1 = centerX + radius * Math.cos(rad);
    const y1 = centerY + radius * Math.sin(rad);
    const x2 = centerX + (radius - 5) * Math.cos(rad);
    const y2 = centerY + (radius - 5) * Math.sin(rad);
    const tx = centerX + (radius + 10) * Math.cos(rad);
    const ty = centerY + (radius + 10) * Math.sin(rad);

    return (
      <g key={`g-tick-${gValue}`}>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="white" strokeWidth="1" />
        <text x={tx} y={ty} fill="white" fontSize="8" textAnchor="middle" dominantBaseline="middle">
          {gValue}
        </text>
      </g>
    );
  });

  return (
    <div className="g-meter-container">
      <svg viewBox="0 0 100 70" width="100%" height="100%">
        {/* Arc path for the scale */}
        <path
          d={`
            M ${centerX + radius * Math.cos(startAngle * Math.PI / 180)} ${centerY + radius * Math.sin(startAngle * Math.PI / 180)}
            A ${radius} ${radius} 0 1 1 ${centerX + radius * Math.cos(endAngle * Math.PI / 180)} ${centerY + radius * Math.sin(endAngle * Math.PI / 180)}
          `}
          stroke="white"
          strokeWidth="2"
          fill="none"
        />

        {/* Render the tick marks */}
        {ticks}

        {/* The Needle */}
        <g transform={`rotate(${rotation} ${centerX} ${centerY})`}>
          <polygon points={`${centerX},${centerY-2} ${centerX},${centerY+2} ${centerX + needleLength},${centerY}`} fill="#00FF00" />
        </g>
        
        {/* Center hub */}
        <circle cx={centerX} cy={centerY} r="3" fill="white" />
        
        {/* Digital Readout Box */}
        <rect x="35" y="55" width="30" height="15" fill="#000" stroke="white" strokeWidth="1" />
        <text x="50" y="63" fill="#00FF00" fontSize="8" textAnchor="middle">
          {g.toFixed(1)}g
        </text>
        <text x="50" y="50" fill="white" fontSize="6" textAnchor="middle">Accel</text>
      </svg>
    </div>
  );
};

export default GForceMeter;