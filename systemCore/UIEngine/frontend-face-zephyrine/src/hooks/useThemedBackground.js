// src/hooks/useThemedBackground.js
import { useEffect } from 'react';

/**
 * Manages the main background container reference.
 * The themed background elements (clouds or stars) are now conditionally rendered directly in App.jsx.
 * @param {React.RefObject<HTMLDivElement>} backgroundRef - A ref to the container div for the background.
 */
// REMOVED: starsCanvasRef from arguments, and all canvas-related logic
export function useThemedBackground(backgroundRef) {
  // This hook no longer handles direct canvas drawing or cleanup.
  // Its purpose is primarily to hold the ref for the background container.
  // Any specific background animations (like stars or clouds) are now
  // handled by other components/CSS based on the 'theme' class.

  // Optional: If you had any styling applied directly by JS to backgroundRef.current,
  // it would go here. Otherwise, this hook primarily serves to hold the ref and
  // signal that the background setup is handled.
  useEffect(() => {
    // This effect might be empty if all background logic is handled by CSS/conditional rendering.
    // Or it could be used for very high-level background setup if needed.
    // For now, it simply ensures the ref is used.
  }, [backgroundRef]);
}