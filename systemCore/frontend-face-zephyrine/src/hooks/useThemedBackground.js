// src/hooks/useThemedBackground.js
import { useEffect, useRef, useCallback } from 'react';

/**
 * Manages the animated background for the application, switching between
 * a starfield for the dark theme and a sky with clouds for the light theme.
 * @param {React.RefObject<HTMLDivElement>} backgroundRef - A ref to the container div for the background.
 * @param {React.RefObject<HTMLCanvasElement>} starsCanvasRef - A ref to the canvas element for stars.
 */
export function useThemedBackground(backgroundRef, starsCanvasRef) {
  // We no longer directly depend on `theme` here, as `App.jsx` conditionally renders the canvas.
  // The effect will re-run when `starsCanvasRef.current` becomes available or unavailable.
  const animationFrameId = useRef(null);

  const cleanupCanvas = useCallback(() => {
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }
    // No longer removing canvas element; React handles its lifecycle.
    const canvas = starsCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear content
    }
  }, [starsCanvasRef]); // Dependency on starsCanvasRef

  useEffect(() => {
    const canvas = starsCanvasRef.current;

    // If canvas is null (e.g., in light mode, or unmounted), cleanup and return
    if (!canvas) {
      cleanupCanvas();
      return;
    }

    // If canvas exists (i.e., in dark mode)
    const ctx = canvas.getContext('2d');
    let stars = [];
    const numStars = 200; // Reduced default number of stars for performance

    const setup = () => {
      // Ensure canvas is attached to DOM before setting width/height
      if (!canvas.parentElement) {
          // This case should ideally not happen now that React renders the canvas
          console.warn("Canvas parent is null during setup, skipping resize.");
          return;
      }
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      stars = []; // Re-initialize stars on resize
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
      if (!canvas.parentElement) { // Check if canvas is still in DOM
        // If canvas is removed by React, stop animation
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
        return;
      }
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

      animationFrameId.current = requestAnimationFrame(draw);
    };

    setup(); // Initial setup
    draw();  // Start drawing

    window.addEventListener('resize', setup); // Re-setup on window resize

    // Cleanup function for this useEffect
    return () => {
      window.removeEventListener('resize', setup);
      cleanupCanvas(); // Call our defined cleanup
    };
  }, [starsCanvasRef, cleanupCanvas]); // Dependencies: starsCanvasRef ensures effect re-runs when canvas element changes (mounted/unmounted)

  // useThemedBackground no longer returns stars array as it's not directly needed by App.jsx
}