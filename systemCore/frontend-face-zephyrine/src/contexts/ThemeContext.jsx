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
    const savedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    // A. Check if a valid theme is explicitly saved in localStorage
    if (savedTheme === 'dark' || savedTheme === 'light') {
      return savedTheme; // Use the user's saved preference
    }

    // B. If no valid saved theme, check the system preference
    return systemPrefersDark ? 'dark' : 'light'; // Use system preference
  });

  useEffect(() => {
    const root = window.document.documentElement;
    // Remove both classes first to avoid potential conflicts
    root.classList.remove('light', 'dark');
    // Add the current theme class
    root.classList.add(theme);
    localStorage.setItem('theme', theme); // Always save the current theme to localStorage
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
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
