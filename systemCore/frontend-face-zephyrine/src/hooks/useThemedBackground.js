// src/hooks/useThemedBackground.js
import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';

/**
 * Manages the animated background, switching between stars (dark) and clouds (light).
 * Animation is high-frequency on the homepage ("/") and low-frequency on all other pages.
 * @param {React.RefObject<HTMLDivElement>} backgroundRef - A ref to the container div for the background.
 */
export function useThemedBackground(backgroundRef) {
  const { theme } = useTheme();
  const location = useLocation(); // Hook to get the current URL path
  const animationFrameId = useRef(null);
  const intervalId = useRef(null);

  useEffect(() => {
    const backgroundContainer = backgroundRef.current;
    if (!backgroundContainer) return;

    const isHomePage = location.pathname === '/';
    const updateRate = isHomePage ? 16 : 10000; // ~60fps vs. every 10 seconds

    // Cleanup function to clear any running animations/intervals
    const cleanup = () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
      }
      if (intervalId.current) {
        clearInterval(intervalId.current);
        intervalId.current = null;
      }
      // Clear previous canvas/clouds
      while (backgroundContainer.firstChild) {
        backgroundContainer.removeChild(backgroundContainer.firstChild);
      }
    };

    cleanup(); // Clean up before setting up the new theme/rate

    // --- Dark Theme Logic: Starfield Canvas ---
    if (theme === 'dark') {
      const canvas = document.createElement('canvas');
      backgroundContainer.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      let stars = [];
      const numStars = isHomePage ? 200 : 75; // Fewer stars on other pages

      const setupStars = () => {
        if (!canvas.parentElement) return;
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        stars = [];
        for (let i = 0; i < numStars; i++) {
          stars.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            radius: Math.random() * 1.5,
            alpha: Math.random() * 0.5 + 0.2,
            speed: (Math.random() * 0.5 + 0.1) * (isHomePage ? 1 : 0.1),
          });
        }
      };

      const drawStars = () => {
        if (!canvas.parentElement || !ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = `rgba(255, 255, 255, 0.8)`;
        
        stars.forEach(star => {
          star.y -= star.speed;
          if (star.y < 0) {
            star.y = canvas.height;
          }
          ctx.beginPath();
          ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
          ctx.fill();
        });
      };

      const animateStars = () => {
        drawStars();
        if (isHomePage) {
          animationFrameId.current = requestAnimationFrame(animateStars);
        }
      };
      
      setupStars();
      if (isHomePage) {
        animateStars();
      } else {
        drawStars(); // Initial draw
        intervalId.current = setInterval(drawStars, updateRate);
      }
      
      window.addEventListener('resize', setupStars);
      return () => window.removeEventListener('resize', setupStars);

    // --- Light Theme Logic: Cloud DOM Elements ---
    } else {
      const cloudsContainer = document.createElement('div');
      cloudsContainer.className = 'clouds';
      backgroundContainer.appendChild(cloudsContainer);

      const clouds = [
        { el: document.createElement('div'), x: 150, speed: isHomePage ? 0.1 : 0.01 },
        { el: document.createElement('div'), x: 400, speed: isHomePage ? 0.05 : 0.005 },
        { el: document.createElement('div'), x: 800, speed: isHomePage ? 0.08 : 0.008 },
      ];

      clouds.forEach((cloud, i) => {
        cloud.el.className = `cloud c${i + 1}`;
        cloudsContainer.appendChild(cloud.el);
      });

      const moveClouds = () => {
        const viewportWidth = window.innerWidth;
        clouds.forEach(cloud => {
          cloud.x += cloud.speed;
          if (cloud.x > viewportWidth + 400) {
            cloud.x = -400;
          }
          cloud.el.style.transform = `translateX(${cloud.x}px)`;
        });
      };
      
      const animateClouds = () => {
          moveClouds();
          if(isHomePage) {
            animationFrameId.current = requestAnimationFrame(animateClouds);
          }
      }

      if (isHomePage) {
        animateClouds();
      } else {
        moveClouds(); // Initial position
        intervalId.current = setInterval(moveClouds, updateRate);
      }
    }

    // Main cleanup function for the entire effect
    return cleanup;

  }, [theme, location.pathname, backgroundRef]); // Re-run effect if theme or path changes
}