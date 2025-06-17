// externalAnalyzer/frontend-face-zephyrine/src/hooks/useStarBackground.js
import { useState, useEffect, useCallback, useRef } from 'react';

const NUM_STARS = 80;
const UPDATE_INTERVAL = 5 * 60 * 1000;
const BATCH_UPDATE_DURATION = 2000;
const NUM_BATCHES = 10;

export const useStarBackground = () => {
  const [stars, setStars] = useState([]);
  const animationFrameId = useRef(null);
  const currentBatchIndex = useRef(0);
  const newStarDataRef = useRef([]);

  const generateSingleStar = useCallback(() => {
    const size = Math.random() * 2 + 1; // Range: 1 to 3 pixels for star size
    // NEW: Generate a random animation duration between 10 and 30 seconds
    const duration = Math.random() * 20 + 10; // Range: 10 to 30 seconds
    return {
      id: Math.random().toString(36).substring(2, 9) + Math.random().toString(36).substring(2, 9),
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      width: `${size}px`,
      height: `${size}px`,
      animationDelay: Math.random() * duration, // Stagger delay based on individual star's duration
      animationDuration: `${duration}s`, // Pass the full duration as a string with 's'
    };
  }, []);

  const processStarBatches = useCallback(() => {
    if (currentBatchIndex.current < NUM_BATCHES) {
      const batchSize = Math.ceil(NUM_STARS / NUM_BATCHES);
      const startIndex = currentBatchIndex.current * batchSize;
      const endIndex = Math.min(startIndex + batchSize, NUM_STARS);
      
      const starsToUpdateInThisBatch = newStarDataRef.current.slice(startIndex, endIndex);

      setStars(prevStars => {
        const updatedStars = [...prevStars];
        for (let i = 0; i < starsToUpdateInThisBatch.length; i++) {
          if (startIndex + i < updatedStars.length) {
            updatedStars[startIndex + i] = starsToUpdateInThisBatch[i];
          } else {
            updatedStars.push(starsToUpdateInThisBatch[i]);
          }
        }
        if (newStarDataRef.current.length < updatedStars.length && currentBatchIndex.current === NUM_BATCHES -1) {
            return updatedStars.slice(0, newStarDataRef.current.length);
        }
        return updatedStars;
      });

      currentBatchIndex.current++;
      animationFrameId.current = requestAnimationFrame(processStarBatches);
    } else {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
      }
      console.log('Star background batch update complete.');
    }
  }, []);

  const regenerateAndBatchUpdateStars = useCallback(() => {
    console.log('Regenerating star background (batched update starting)...');
    newStarDataRef.current = Array.from({ length: NUM_STARS }, generateSingleStar);
    
    if (stars.length === 0) {
        setStars(newStarDataRef.current);
        console.log('Initial stars set.');
        return;
    }

    currentBatchIndex.current = 0;
    
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
    }
    animationFrameId.current = requestAnimationFrame(processStarBatches);
  }, [generateSingleStar, processStarBatches, stars.length]);

  useEffect(() => {
    const initialStars = Array.from({ length: NUM_STARS }, generateSingleStar);
    setStars(initialStars);
    console.log('Initial star background generated.');

    const intervalId = setInterval(() => {
      regenerateAndBatchUpdateStars();
    }, UPDATE_INTERVAL);

    return () => {
      clearInterval(intervalId);
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
      console.log('Star background update interval and animation frame cleared.');
    };
  }, [generateSingleStar, regenerateAndBatchUpdateStars, UPDATE_INTERVAL]);

  return stars;
};