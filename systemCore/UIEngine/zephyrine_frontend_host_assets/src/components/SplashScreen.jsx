// src/components/SplashScreen.jsx
import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import '../styles/components/_splashScreen.css';

const SplashScreen = ({ isVisible, onFadeOutComplete }) => {
  const [isOverlayFadingOut, setIsOverlayFadingOut] = useState(false);
  
  // State for the logo image source and its specific animation class
  const [currentLogo, setCurrentLogo] = useState('/img/zephyrineFoundation.png');
  const [logoClass, setLogoClass] = useState('logo-fade-in');

  useEffect(() => {
    let timers = [];

    if (isVisible) {
      // --- Sequence Start ---
      
      // 1. First Logo (Foundation) is set by default state.
      // Wait 2.0s, then fade out Foundation Logo
      timers.push(setTimeout(() => {
        setLogoClass('logo-fade-out');
      }, 2000));

      // 2. Switch to Project Logo and Fade In
      timers.push(setTimeout(() => {
        setCurrentLogo('/img/ProjectZephy023LogoRenewal.png');
        setLogoClass('logo-fade-in');
      }, 3000));

      // 3. Start final Overlay fade out
      timers.push(setTimeout(() => {
        setIsOverlayFadingOut(true);
      }, 4000));
    }

    return () => timers.forEach(timer => clearTimeout(timer));
  }, [isVisible]);

  const handleOverlayAnimationEnd = (e) => {
    // Check if it's the overlay fading out (opacity transition)
    if (e.propertyName === 'opacity' && e.target.classList.contains('splash-screen-overlay')) {
      onFadeOutComplete();
    }
  };

  return (
    <div 
      className={`splash-screen-overlay ${isVisible ? 'visible' : ''} ${isOverlayFadingOut ? 'fade-out' : ''}`}
      onTransitionEnd={handleOverlayAnimationEnd}
    >
      <div className="splash-screen-content">
        <img
          src={currentLogo}
          alt="Splash Screen Logo"
          className={`splash-screen-logo ${logoClass}`}
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