// src/components/shuttle-displays/HorizontalSituationIndicatorPage.jsx
import React from 'react';
import '../../styles/components/_hsiDisplay.css';

const toRad = (deg) => deg * (Math.PI / 180);

// BearingPointer component can remain the same
const BearingPointer = ({ bearing, label, shape = 'triangle' }) => {
    const pointerShapes = {
        triangle: <polygon points="0,-8 6,8 -6,8" />,
        asterisk: <path d="M-5,-5 L5,5 M-5,5 L5,-5 M-7,0 L7,0 M0,-7 L0,7" strokeWidth="2" />,
        c: <text y="4" textAnchor="middle" fontSize="16px">C</text>,
        e: <text y="4" textAnchor="middle" fontSize="16px">E</text>,
        i: <text y="4" textAnchor="middle" fontSize="16px">I</text>
    };
    const finalShape = (shape || '').toLowerCase();
    return (
        <g transform={`rotate(${bearing})`} className={`bearing-pointer shape-${finalShape}`}>
        {pointerShapes[finalShape] || pointerShapes.triangle}
        </g>
    );
};


const HorizontalSituationIndicatorPage = ({ data }) => {
  const heading = data?.flight_dynamics?.attitude?.yaw || 0;
  const hsi = data?.navigation?.hsi || {};
  const {
    selected_course = 0,
    course_deviation = 10,
    hac_turn_angle,
    waypoints = []
  } = hsi;
  
  const runwayWaypoint = waypoints.find(wp => wp.label === '*');
  const hacCenterWaypoint = waypoints.find(wp => wp.label === 'C');

  const cdi_deflection = Math.max(-45, Math.min(45, course_deviation * 10));

  const size = 500;
  const center = size / 2;
  const compassRadius = size / 2 - 40;

  // --- THIS IS THE VALUE TO MODIFY ---
  // Controls the radius of the circle where the labels are placed.
  // Increase it to move labels further out, decrease to move them in.
  const labelRadius = compassRadius + 20;


  const compassTicks = [];
  for (let i = 0; i < 360; i += 30) {
    const label = String(i / 10).padStart(2, '0');
    
    // Calculate the position using the new labelRadius
    const angleRad = toRad(i);
    const textX = center + labelRadius * Math.sin(angleRad);
    const textY = center - labelRadius * Math.cos(angleRad);
    
    compassTicks.push(
      <g key={`tick-${i}`}>
        {/* The line rotates with its group */}
        <g transform={`rotate(${i} ${center} ${center})`}>
          <line x1={center} y1={40} x2={center} y2={55} className="compass-tick-line" />
        </g>
        {/* The text is positioned absolutely and then counter-rotated */}
        <text 
          x={textX} 
          y={textY}
          transform={`rotate(${i} ${textX} ${textY})`}
          className="compass-tick-label"
          textAnchor="middle" 
          dominantBaseline="middle"
        >
          {label}
        </text>
      </g>
    );
  }

  return (
    <div className="hsi-container">
      <svg viewBox={`0 0 ${size} ${size}`} className="hsi-svg">
        <g transform={`translate(${center}, ${center})`}>
          <g transform={`rotate(${-heading})`}>
            <circle r={compassRadius} className="compass-card-bg" />
            <g>{compassTicks}</g>
            
            {waypoints.map(wp => (
              <BearingPointer key={wp.id} bearing={wp.bearing} label={wp.label} shape={wp.label} />
            ))}
             {waypoints.map(wp => (
              <BearingPointer key={`${wp.id}-recip`} bearing={wp.bearing + 180} label={wp.label} shape={wp.label} />
            ))}
          </g>

          <g className="hsi-static-elements">
            <line y1={-compassRadius - 10} y2={-compassRadius} className="lubber-line" />
            <polygon points="0,-20 -15,0 15,0" className="fixed-aircraft-symbol" />
            
            <g className="course-deviation-scale">
                <circle cx="-40" r="4" /> <circle cx="-20" r="4" />
                <circle cx="20" r="4" />  <circle cx="40" r="4" />
            </g>
            
            <g transform={`rotate(${selected_course - heading})`}>
              <path d={`M0,-${compassRadius-10} L0,${compassRadius-10}`} className="course-arrow-line" />
              <polygon points="0,-180 -20,-160 20,-160" className="course-arrow-head" />
              <g transform={`translate(${cdi_deflection}, 0)`}>
                <rect x="-25" y="-5" width="50" height="10" className="cdi-flag-box" />
                <text x="0" y="4" className="cdi-flag-text">CDI</text>
                <path d={`M0,-90 L0,90`} className="cdi-needle" />
              </g>
            </g>
          </g>
        </g>
      </svg>
      
      <div className="hsi-data-overlay">
        <div className="hsi-data-box turn-angle">
            <span className="label">ΔAz</span>
            <span className="value">{hac_turn_angle !== undefined ? `${hac_turn_angle.toFixed(0)}°` : 'N/A'}</span>
        </div>
        <div className="hsi-data-box range-runway">
            <span className="label">{runwayWaypoint?.label || '*'}</span>
            <span className="value">{runwayWaypoint?.range || '----'}</span>
        </div>
         <div className="hsi-data-box range-hac">
            <span className="label">{hacCenterWaypoint?.label || 'HAC-C'}</span>
            <span className="value">{hacCenterWaypoint?.range || '----'}</span>
        </div>
      </div>
    </div>
  );
};

export default HorizontalSituationIndicatorPage;