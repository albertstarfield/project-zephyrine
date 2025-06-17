// src/components/SplashScreen.jsx
import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import '../styles/components/_splashScreen.css';

const SplashScreen = ({ isVisible, onFadeOutComplete }) => {
  const [isFadingOut, setIsFadingOut] = useState(false);

  useEffect(() => {
    let fadeOutTimer;
    if (isVisible) {
      fadeOutTimer = setTimeout(() => {
        setIsFadingOut(true);
      }, 2500); // Wait 2.5s to start fading out
    }
    return () => clearTimeout(fadeOutTimer);
  }, [isVisible]);

  const handleOverlayAnimationEnd = (e) => {
    if (e.propertyName === 'opacity' && e.target.classList.contains('splash-screen-overlay')) {
      onFadeOutComplete();
    }
  };

  return (
    <div className={`splash-screen-overlay ${isVisible ? 'visible' : ''} ${isFadingOut ? 'fade-out' : ''}`}
         onTransitionEnd={handleOverlayAnimationEnd}>
      <div className="splash-screen-content">
        <img
          src="/img/ProjectZephy023LogoRenewal.png"
          alt="Project Zephyrine Logo"
          className="splash-screen-logo"
        />
      </div>
    </div>
  );
};

SplashScreen.propTypes = {
  isVisible: PropTypes.bool.isRequired,
  onFadeOutComplete: PropTypes.func.isRequired,
};

export default SplashScreen;