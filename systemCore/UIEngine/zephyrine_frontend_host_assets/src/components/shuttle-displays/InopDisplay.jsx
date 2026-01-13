// src/components/shuttle-displays/InopDisplay.jsx
import React from 'react';

const InopDisplay = ({ title }) => {
  return (
    <div className="inop-container">
      <h1>{title}</h1>
      <div className="inop-box">INOP</div>
    </div>
  );
};

export default InopDisplay;