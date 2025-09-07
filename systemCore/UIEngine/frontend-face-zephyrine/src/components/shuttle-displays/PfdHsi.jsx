// src/components/shuttle-displays/PfdHsi.jsx
import React from 'react';

const PfdHsi = ({ heading, course }) => {
  // The outer SVG container can be of any size, viewBox handles scaling
  const size = 200;
  const center = size / 2;
  const radius = size / 2 - 10;

  // Generate the compass ticks and labels
  const compassTicks = [];
  for (let i = 0; i < 360; i += 10) {
    const isMajorTick = i % 30 === 0;
    const tickLength = isMajorTick ? 10 : 5;
    const label = (i === 0) ? 'N' : (i === 90) ? 'E' : (i === 180) ? 'S' : (i === 270) ? 'W' : i / 10;
    
    compassTicks.push(
      <g key={`tick-${i}`} transform={`rotate(${i} ${center} ${center})`}>
        <line
          x1={center}
          y1={10}
          x2={center}
          y2={10 + tickLength}
          stroke="white"
          strokeWidth={isMajorTick ? "2" : "1"}
        />
        {isMajorTick && (
          <text
            x={center}
            y={30}
            fill="white"
            fontSize="12"
            textAnchor="middle"
            transform={`rotate(${-i} ${center} 30)`}
          >
            {label}
          </text>
        )}
      </g>
    );
  }

  return (
    <div className="pfd-hsi-container">
      <svg viewBox={`0 0 ${size} ${size}`} width="100%" height="100%">
        {/* Rotating group for the compass card */}
        <g transform={`rotate(${-heading} ${center} ${center})`}>
          {/* Compass Rose Background */}
          <circle cx={center} cy={center} r={radius} fill="#333" stroke="white" strokeWidth="1" />
          {/* The ticks and labels */}
          {compassTicks}

          {/* Selected Course Pointer (Magenta) */}
          <g transform={`rotate(${course} ${center} ${center})`}>
            <path
              d={`M ${center},${10} L ${center-8},${25} L ${center},${20} L ${center+8},${25} Z`}
              fill="#FF00FF"
            />
          </g>
        </g>
        
        {/* Fixed Aircraft Symbol and Heading Pointer */}
        <g>
          {/* Main heading triangle at the top */}
          <polygon points={`${center},10 ${center-6},20 ${center+6},20`} fill="white" />
          {/* Little aircraft symbol */}
          <path d={`M ${center-5},${center+5} L ${center},${center-10} L ${center+5},${center+5} Z`} fill="#555" stroke="white"/>
        </g>

        {/* Deviation dots would go here, static relative to the fixed aircraft */}
         <circle cx={center - 20} cy={center} r="2" fill="white" />
         <circle cx={center + 20} cy={center} r="2" fill="white" />

      </svg>
    </div>
  );
};

export default PfdHsi;