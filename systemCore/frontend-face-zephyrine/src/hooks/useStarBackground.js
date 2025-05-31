// ExternalAnalyzer/frontend-face-zephyrine/src/hooks/useStarBackground.js
import { useState, useEffect, useCallback, useRef } from 'react';

const NUM_STARS = 80; // Reduced default number of stars for performance
const UPDATE_INTERVAL = 5 * 60 * 1000; // 5 minutes in milliseconds
const BATCH_UPDATE_DURATION = 2000; // Duration over which to apply batched updates (e.g., 2 seconds)
const NUM_BATCHES = 10; // Number of batches to split the update into

// Changed to a named export
export const useStarBackground = () => {
  const [stars, setStars] = useState([]);
  const animationFrameId = useRef(null); // For managing batched updates
  const currentBatchIndex = useRef(0);
  const newStarDataRef = useRef([]);

  const generateSingleStar = useCallback(() => {
    const size = Math.random() * 2 + 0.8; // Star size between 0.8px and 2.8px
    return {
      id: Math.random().toString(36).substring(2, 9) + Math.random().toString(36).substring(2, 9),
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      width: `${size}px`,
      height: `${size}px`,
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
        // Replace stars in the current batch range
        for (let i = 0; i < starsToUpdateInThisBatch.length; i++) {
          if (startIndex + i < updatedStars.length) {
            updatedStars[startIndex + i] = starsToUpdateInThisBatch[i];
          } else {
            // This case should ideally not happen if prevStars is initialized correctly
            updatedStars.push(starsToUpdateInThisBatch[i]);
          }
        }
        // If newStarDataRef.current is shorter than prevStars (e.g. NUM_STARS decreased), trim prevStars
        if (newStarDataRef.current.length < updatedStars.length && currentBatchIndex.current === NUM_BATCHES -1) {
            return updatedStars.slice(0, newStarDataRef.current.length);
        }
        return updatedStars;
      });

      currentBatchIndex.current++;
      animationFrameId.current = requestAnimationFrame(processStarBatches);
    } else {
      // All batches processed
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
    
    // If stars haven't been initialized yet, set them all at once
    if (stars.length === 0) {
        setStars(newStarDataRef.current);
        console.log('Initial stars set.');
        return;
    }

    currentBatchIndex.current = 0; // Reset batch counter
    
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current); // Cancel any ongoing batch update
    }
    animationFrameId.current = requestAnimationFrame(processStarBatches);
  }, [generateSingleStar, processStarBatches, stars.length]);


  useEffect(() => {
    // Generate the initial set of stars (all at once for the first paint)
    const initialStars = Array.from({ length: NUM_STARS }, generateSingleStar);
    setStars(initialStars);
    console.log('Initial star background generated.');

    // Set up an interval to regenerate the stars periodically
    const intervalId = setInterval(() => {
      regenerateAndBatchUpdateStars();
    }, UPDATE_INTERVAL);

    // Cleanup function
    return () => {
      clearInterval(intervalId);
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
      console.log('Star background update interval and animation frame cleared.');
    };
  }, [generateSingleStar, regenerateAndBatchUpdateStars, UPDATE_INTERVAL]); // Dependencies

  return stars;
};
