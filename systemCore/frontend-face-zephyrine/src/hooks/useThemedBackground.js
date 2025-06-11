// src/hooks/useThemedBackground.js
import { useEffect } from 'react';
import { useTheme } from '../contexts/ThemeContext';

/**
 * Manages the animated background for the application, switching between
 * a starfield for the dark theme and a sky with clouds for the light theme.
 * @param {React.RefObject<HTMLDivElement>} backgroundRef - A ref to the container div for the background.
 */
export function useThemedBackground(backgroundRef) {
  const { theme } = useTheme();

  useEffect(() => {
    const backgroundContainer = backgroundRef.current;
    if (!backgroundContainer) return;

    let animationFrameId;

    const cleanup = () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      const existingCanvas = backgroundContainer.querySelector('canvas');
      if (existingCanvas) {
        existingCanvas.remove();
      }
    };

    // If the theme is not dark, ensure everything is cleaned up.
    if (theme !== 'dark') {
      cleanup();
      return; // Stop here for light theme.
    }

    // --- Dark Theme Logic: Create and animate the starfield ---

    // Prevent creating a new canvas if one already exists
    if (backgroundContainer.querySelector('canvas')) {
      return;
    }

    const canvas = document.createElement('canvas');
    backgroundContainer.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    let stars = [];
    const numStars = 200;

    const setup = () => {
      if (!canvas.parentElement) return; // Stop if canvas was removed
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      stars = [];
      for (let i = 0; i < numStars; i++) {
        stars.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          radius: Math.random() * 1.5,
          alpha: Math.random(),
          speed: Math.random() * 0.5 + 0.1,
        });
      }
    };

    const draw = () => {
      if (!canvas.parentElement) return; // Stop drawing if canvas was removed
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      stars.forEach(star => {
        star.y -= star.speed;
        if (star.y < 0) {
          star.y = canvas.height;
        }

        ctx.beginPath();
        ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${star.alpha})`;
        ctx.fill();
      });

      animationFrameId = requestAnimationFrame(draw);
    };

    setup();
    draw();

    window.addEventListener('resize', setup);

    // This cleanup function will run when the theme changes from dark to light,
    // or when the component unmounts.
    return () => {
      window.removeEventListener('resize', setup);
      cleanup();
    };
  }, [theme, backgroundRef]); // Rerun this effect whenever the theme changes.
}
