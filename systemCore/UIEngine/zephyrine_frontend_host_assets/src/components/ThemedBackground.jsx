import React, { useState, useEffect, useRef } from 'react';
import { useTheme } from '../contexts/ThemeContext';
import { useStarBackground } from '../hooks/useStarBackground';
import StarParticle from './StarParticle';
import html2canvas from 'html2canvas';

// Simple cloud shapes for the night (reusing structure or creating new)
const NightClouds = () => (
  <div className="night-cloud-container">
    <div className="night-cloud night-cloud-1"></div>
    <div className="night-cloud night-cloud-2"></div>
    <div className="night-cloud night-cloud-3"></div>
  </div>
);

const ThemedBackground = () => {
  const { theme } = useTheme();
  const stars = useStarBackground();
  
  // Ref to capture the container
  const containerRef = useRef(null);
  // State to store the captured static image
  const [frozenBackground, setFrozenBackground] = useState(null);

  useEffect(() => {
    // 1. When theme changes, unfreeze immediately to play animation
    setFrozenBackground(null);

    // 2. Set a timer to capture and freeze the background after 5 seconds
    const freezeTimer = setTimeout(async () => {
      if (containerRef.current) {
        try {
          // Capture the background container
          const canvas = await html2canvas(containerRef.current, {
            scale: 1, // Keep 1:1 scale for performance; increase to window.devicePixelRatio for higher quality
            logging: false,
            useCORS: true,
            backgroundColor: null, // Preserves the CSS gradient transparency
          });

          // Convert canvas to image data URL
          const imgData = canvas.toDataURL('image/png');
          setFrozenBackground(imgData);
          
        } catch (error) {
          console.error("Background freeze failed:", error);
          // If capture fails, we just stay in animated mode
        }
      }
    }, 5000); // 5 seconds duration

    // Cleanup timer on unmount or theme change
    return () => clearTimeout(freezeTimer);
  }, [theme]);

  // If frozen, apply the captured image as the background style
  const containerStyle = frozenBackground 
    ? { 
        backgroundImage: `url(${frozenBackground})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        // Overwrite the CSS gradient because it's baked into the image now
        background: `url(${frozenBackground}) no-repeat center center / cover`
      }
    : {};

  return (
    <div 
      className="background-container" 
      ref={containerRef}
      style={containerStyle}
    >
      {/* Only render the heavy animated elements if we haven't frozen the background yet */}
      {!frozenBackground && (
        <>
          {theme === 'light' && (
            <div className="clouds">
              <div className="cloud c1"></div>
              <div className="cloud c2"></div>
              <div className="cloud c3"></div>
            </div>
          )}
          {theme === 'dark' && (
            <div className="svg-star-background">
              {/* Optional: Include NightClouds if you want them rendered before capture */}
              {/* <NightClouds /> */} 
              
              {stars.map((star) => (
                <StarParticle
                  key={star.id}
                  id={star.id}
                  top={star.top}
                  left={star.left}
                  width={star.width}
                  height={star.height}
                  animationDelay={star.animationDelay}
                  animationDuration={star.animationDuration}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

// Memoize the component to prevent re-renders when the parent (App) changes state.
export default React.memo(ThemedBackground);