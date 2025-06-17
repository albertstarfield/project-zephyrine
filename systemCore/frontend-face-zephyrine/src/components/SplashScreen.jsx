// src/components/SplashScreen.jsx
import React, { useState } from 'react';
import PropTypes from 'prop-types';
import '../styles/components/_splashScreen.css';

const SplashScreen = ({ isVisible, onAnimationEnd }) => {
  const [logoLoaded, setLogoLoaded] = useState(false);

  const handleLogoLoad = () => {
    setLogoLoaded(true);
  };

  return (
    <div className={`splash-screen-overlay ${isVisible ? 'visible' : ''}`}>
      <div className={`splash-screen-content ${logoLoaded ? 'loaded' : ''}`}> {/* Add 'loaded' class */}
        <img
          src="/img/ProjectZephy023LogoRenewal.png"
          alt="Project Zephyrine Logo"
          className="splash-screen-logo"
          onLoad={handleLogoLoad} // NEW: Add onLoad handler
        />
      </div>
    </div>
  );
};

SplashScreen.propTypes = {
  isVisible: PropTypes.bool.isRequired,
  onAnimationEnd: PropTypes.func.isRequired,
};

export default SplashScreen;