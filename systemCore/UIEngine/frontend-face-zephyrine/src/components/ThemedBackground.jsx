import React from 'react';
import { useTheme } from '../contexts/ThemeContext';
import { useStarBackground } from '../hooks/useStarBackground';
import StarParticle from './StarParticle';

const ThemedBackground = () => {
  const { theme } = useTheme();
  const stars = useStarBackground();

  return (
    <div className="background-container">
      {theme === 'light' && (
        <div className="clouds">
          <div className="cloud c1"></div>
          <div className="cloud c2"></div>
          <div className="cloud c3"></div>
        </div>
      )}
      {theme === 'dark' && (
        <div className="svg-star-background">
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
    </div>
  );
};

// Memoize the component to prevent re-renders when the parent (App) changes state.
export default React.memo(ThemedBackground);