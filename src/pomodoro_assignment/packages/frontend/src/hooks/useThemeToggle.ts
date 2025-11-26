import { useAppSelector, useAppDispatch } from './redux';
import { uiSelectors, toggleTheme, setTheme } from '../store/slices/uiSlice';

export const useThemeToggle = () => {
  const dispatch = useAppDispatch();
  const theme = useAppSelector(uiSelectors.selectTheme);
  const isDarkMode = useAppSelector(uiSelectors.selectIsDarkMode);

  const handleToggleTheme = () => {
    dispatch(toggleTheme());
  };

  const handleSetTheme = (newTheme: 'light' | 'dark' | 'auto') => {
    dispatch(setTheme(newTheme));
  };

  const setLightMode = () => {
    dispatch(setTheme('light'));
  };

  const setDarkMode = () => {
    dispatch(setTheme('dark'));
  };

  const setAutoMode = () => {
    dispatch(setTheme('auto'));
  };

  return {
    // Current state
    theme,
    isDarkMode,
    isLightMode: theme === 'light',
    isAutoMode: theme === 'auto',

    // Actions
    toggleTheme: handleToggleTheme,
    setTheme: handleSetTheme,
    setLightMode,
    setDarkMode,
    setAutoMode,
  };
};