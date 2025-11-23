import React from 'react';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { ThemeProvider } from 'styled-components';
import { store, persistor } from './store';
import { ZenTheme } from './themes';
import { Router } from './components';
import './App.css';

function App() {
  return (
    <Provider store={store}>
      <PersistGate loading={null} persistor={persistor}>
        <ThemeProvider theme={ZenTheme}>
          <Router />
        </ThemeProvider>
      </PersistGate>
    </Provider>
  );
}

export default App;