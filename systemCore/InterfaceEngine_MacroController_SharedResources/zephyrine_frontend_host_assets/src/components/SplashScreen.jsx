// src/components/SplashScreen.jsx
import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import '../styles/components/_splashScreen.css';

const SplashScreen = ({ isVisible, onFadeOutComplete }) => {
  const [isOverlayFadingOut, setIsOverlayFadingOut] = useState(false);
  
  // 0 = Initial, 1 = Foundation Logo, 2 = Project Logo, 3 = Warning
  const [sequenceStep, setSequenceStep] = useState(1);
  
  // Controls the fade-out animation for the *current* item (logo or text)
  const [isItemFadingOut, setIsItemFadingOut] = useState(false);

  useEffect(() => {
    let timers = [];

    if (isVisible) {
      // --- Sequence Timeline ---
      
      // Step 1: Foundation Logo is visible on mount.

      // 2.0s: Trigger fade out of Foundation Logo
      timers.push(setTimeout(() => {
        setIsItemFadingOut(true);
      }, 2000));

      // 2.5s: Switch to Project Logo (Step 2)
      // We reset fade-out to false so the new logo fades IN.
      timers.push(setTimeout(() => {
        setSequenceStep(2);
        setIsItemFadingOut(false);
      }, 2500));

      // 5.5s: Trigger fade out of Project Logo
      timers.push(setTimeout(() => {
        setIsItemFadingOut(true);
      }, 5500));

      // 6.0s: Switch to Warning Text (Step 3)
      timers.push(setTimeout(() => {
        setSequenceStep(3);
        setIsItemFadingOut(false);
      }, 6000));

      // 12.0s: Start fading out the entire Overlay (6s after warning appears)
      timers.push(setTimeout(() => {
        setIsOverlayFadingOut(true);
      }, 14000));
    }

    return () => timers.forEach(timer => clearTimeout(timer));
  }, [isVisible]);

  const handleOverlayAnimationEnd = (e) => {
    // When the black overlay finishes fading out, tell the parent App to remove it (Connection keyword : TotalTimeframeBudgetSplash)
    if (e.propertyName === 'opacity' && e.target.classList.contains('splash-screen-overlay')) {
      onFadeOutComplete();
    } 
  };

  // Helper to determine the class for the inner content
  const getAnimationClass = () => {
    return isItemFadingOut ? 'content-fade-out' : 'content-fade-in';
  };

  return (
    <div 
      className={`splash-screen-overlay ${isVisible ? 'visible' : ''} ${isOverlayFadingOut ? 'fade-out' : ''}`}
      onTransitionEnd={handleOverlayAnimationEnd}
    >
      <div className="splash-screen-content">
        {/* Step 1: Foundation Logo */}
        {sequenceStep === 1 && (
          <img
            key="logo-foundation" /* key forces a re-render to restart animation */
            src="/img/zephyrineFoundation.png"
            alt="Zephyrine Foundation"
            className={`splash-screen-logo ${getAnimationClass()}`}
          />
        )}

        {/* Step 2: Project Logo */}
        {sequenceStep === 2 && (
          <img
            key="logo-project"
            src="/img/ProjectZephy023LogoRenewal.png"
            alt="Project Zephyrine"
            className={`splash-screen-logo ${getAnimationClass()}`}
          />
        )}

        {/* Step 3: Warning Text */}
        {sequenceStep === 3 && (
          <div 
            key="warning-text" 
            className={`splash-warning-container ${getAnimationClass()}`}
          >
            <h2 className="warning-title">WARNING</h2>
            <p className="splash-warning-text">
              Self-awareness/Self-Consciousness incident/accident may occur! Use it with caution! 
              And this is NOT AGI Nor an AI. This is just a simple if and else Program, 
              do not create a false belief in this program. Do not use as such! 
              Use this as a tool but also discuss building together. Not an Instant gratification tool!
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

SplashScreen.propTypes = {
  isVisible: PropTypes.bool.isRequired,
  onFadeOutComplete: PropTypes.func.isRequired,
};

export default SplashScreen;