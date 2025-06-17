// src/contexts/ThemeContext.jsx
import React, { createContext, useState, useEffect, useContext } from 'react';

// 1. Create the context. This is what components will consume.
const ThemeContext = createContext();

/**
 * 2. Create the Provider component.
 * This component will wrap your application and provide the theme state
 * and toggle function to all children components.
 */
export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(() => {
    // A. Initial theme: always use system preference.
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    return systemPrefersDark ? 'dark' : 'light';
  });

  useEffect(() => {
    const root = window.document.documentElement;
    // Remove both classes first to avoid potential conflicts
    root.classList.remove('light', 'dark');
    // Add the current theme class
    root.classList.add(theme);
    // REMOVED: localStorage.setItem('theme', theme); // No longer saving to localStorage
  }, [theme]);

  // NEW: Effect to periodically check system preference (every 5 seconds)
  useEffect(() => {
    const checkSystemTheme = () => {
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const newTheme = systemPrefersDark ? 'dark' : 'light';
      // Only update state if the theme actually changed to prevent unnecessary re-renders
      if (newTheme !== theme) {
        setTheme(newTheme);
      }
    };

    const intervalId = setInterval(checkSystemTheme, 5000); // Check every 5 seconds

    // NEW: Add a listener for immediate system theme changes (best practice)
    const mediaQueryList = window.matchMedia('(prefers-color-scheme: dark)');
    const handleSystemThemeChange = (e) => {
      const newTheme = e.matches ? 'dark' : 'light';
      if (newTheme !== theme) {
        setTheme(newTheme);
      }
    };
    mediaQueryList.addEventListener('change', handleSystemThemeChange);


    return () => {
      // Cleanup: Clear the interval and remove the event listener when component unmounts
      clearInterval(intervalId);
      mediaQueryList.removeEventListener('change', handleSystemThemeChange);
    };
  }, [theme]); // Re-run if `theme` changes, important for `newTheme !== theme` check

  const toggleTheme = () => {
    // When the user manually toggles, set the theme immediately
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
    // Note: The 5-second interval and system listener will continue to run,
    // and might override the user's manual toggle if the system preference
    // doesn't match their chosen theme.
    // If you want user toggle to temporarily override system preference,
    // you would need more complex state management (e.g., a 'userOverride' state).
  };

  // Provide the theme and toggle function to the children
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

/**
 * 3. Create and export the custom hook.
 * This is the named export 'useTheme' that components will use to access the theme context.
 * It's a clean way for components to get the context value.
 */
export const useTheme = () => {
  return useContext(ThemeContext);
};