import React from 'react';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { ThemeProvider } from 'styled-components';
import { store, persistor } from './store';
import { ZenTheme } from './themes';
import { Router } from './components';
import { useAppSelector } from './hooks/redux';
import './App.css';

// Theme wrapper component that applies dark mode classes to document
const ThemeWrapper: React.FC = () => {
  const theme = useAppSelector(state => state.ui.theme);

  React.useEffect(() => {
    const isDarkMode = theme === 'dark' ||
      (theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches);

    if (!isDarkMode) {
      document.documentElement.classList.add('dark-mode');
      document.body.classList.add('dark-mode');
    } else {
      document.documentElement.classList.remove('dark-mode');
      document.body.classList.remove('dark-mode');
    }
  }, [theme]);

  return <Router />;
};

function App() {
  return (
    <Provider store={store}>
      <PersistGate loading={null} persistor={persistor}>
        <ThemeProvider theme={ZenTheme}>
          <ThemeWrapper />
        </ThemeProvider>
      </PersistGate>
    </Provider>
  );
}

export default App;