// src/contexts/ThemeContext.jsx
import React, { createContext, useState, useEffect, useContext } from 'react';

const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(() => {
    // Set the initial theme based on the user's system preference
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    return systemPrefersDark ? 'dark' : 'light';
  });

  // Effect 1: Apply the current theme to the HTML document
  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
  }, [theme]);

  // Effect 2: Listen for changes in the user's system theme preference
  useEffect(() => {
    const mediaQueryList = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleSystemThemeChange = (e) => {
      // Update the theme state when the system preference changes
      const newTheme = e.matches ? 'dark' : 'light';
      setTheme(newTheme);
    };

    // Add the event listener
    mediaQueryList.addEventListener('change', handleSystemThemeChange);

    // Cleanup: remove the event listener when the component unmounts
    return () => {
      mediaQueryList.removeEventListener('change', handleSystemThemeChange);
    };
  }, []); // The empty dependency array ensures this effect runs only once

  // The toggle function is no longer needed as the theme is now fully automatic
  // based on system preference, which simplifies the logic.

  return (
    // The value now only needs to provide the theme itself
    <ThemeContext.Provider value={{ theme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  return useContext(ThemeContext);
};