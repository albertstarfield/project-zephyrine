import { useState, useEffect } from 'react';

const STAR_COUNT = 150;

function createStar() {
  return {
    id: Math.random(), // Use random ID for key
    left: `${Math.random() * 100}%`,
    top: `${Math.random() * 100}%`,
    size: `${Math.random() * 2 + 1}px`,
    animationDuration: `${Math.random() * 3 + 2}s`,
    animationDelay: `${Math.random() * 2}s`,
  };
}

export function useStarBackground() {
  const [stars, setStars] = useState([]);

  useEffect(() => {
    const initialStars = Array.from({ length: STAR_COUNT }, createStar);
    setStars(initialStars);
    // No cleanup needed
  }, []); // Run only once on mount

  return stars;
}
