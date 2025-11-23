import React, { createContext, useContext, useEffect, useState } from 'react';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';
import { ZenTheme } from './ZenTheme';

interface ThemeContextType {
  theme: typeof ZenTheme;
  toggleDarkMode?: () => void;
  isDarkMode?: boolean;
}

const ThemeContext = createContext<ThemeContextType>({
  theme: ZenTheme,
});

export const useTheme = () => useContext(ThemeContext);

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check for saved preference or default to light mode
    const saved = localStorage.getItem('theme-mode');
    return saved === 'dark';
  });

  useEffect(() => {
    // Apply dark mode class to document
    if (isDarkMode) {
      document.documentElement.classList.add('dark-mode');
    } else {
      document.documentElement.classList.remove('dark-mode');
    }

    // Save preference to localStorage
    localStorage.setItem('theme-mode', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const toggleDarkMode = () => {
    setIsDarkMode(prev => !prev);
  };

  const theme = {
    ...ZenTheme,
    // Override colors for dark mode if needed
    ...(isDarkMode && {
      colors: {
        ...ZenTheme.colors,
        neutral: {
          ...ZenTheme.colors.neutral,
          50: '#2C3E50',  // Dark background
          100: '#34495E',
          500: '#ECF0F1', // Light text
        },
      },
    }),
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleDarkMode, isDarkMode }}>
      <StyledThemeProvider theme={theme}>
        {children}
      </StyledThemeProvider>
    </ThemeContext.Provider>
  );
};