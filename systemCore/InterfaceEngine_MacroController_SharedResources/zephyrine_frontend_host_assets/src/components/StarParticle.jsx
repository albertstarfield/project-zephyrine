// src/components/StarParticle.jsx
import React, { useState, useEffect } from 'react';

const StarSVG = (props) => (
  <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    {/* Gemini-style 4-pointed 'Plus' Star (Sparkle) */}
    <path
      d="M12 0C12 6.627 17.373 12 24 12C17.373 12 12 17.373 12 24C12 17.373 6.627 12 0 12C6.627 12 12 6.627 12 0Z"
      fill="#ffd885"
    />
  </svg>
);

const StarParticle = ({ id, top, left, animationDelay, width, height, animationDuration }) => {
  const [randomRotation, setRandomRotation] = useState(`${Math.random() * 360 - 180}deg`);

  useEffect(() => {
    let intervalId;

    // After 30 seconds, start updating every 5 seconds.
    const timeoutId = setTimeout(() => {
      intervalId = setInterval(() => {
        setRandomRotation(`${Math.random() * 360 - 180}deg`);
      }, 5000); // Update every 5 seconds
    }, 30000); // Wait 30 seconds

    // Cleanup function to clear timers when the component unmounts
    return () => {
      clearTimeout(timeoutId);
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, []); // Empty dependency array ensures this effect runs only once on mount

  return (
    <div
      key={id}
      className="star-particle-svg"
      style={{
        top,
        left,
        width,
        height,
        animation: `pop-oscillate-fade ${animationDuration} ease-in-out infinite`,
        animationDelay: `${animationDelay}s`,
        '--random-rotation': randomRotation, // Use state for rotation
      }}
    >
      <StarSVG />
    </div>
  );
};

export default React.memo(StarParticle);