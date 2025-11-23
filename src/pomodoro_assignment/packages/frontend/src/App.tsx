import React from 'react';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { ThemeProvider } from 'styled-components';
import { store, persistor } from './store';
import { ZenTheme } from './themes';
import { TimerScreen } from './components';
import './App.css';

function App() {
  return (
    <Provider store={store}>
      <PersistGate loading={null} persistor={persistor}>
        <ThemeProvider theme={ZenTheme}>
          <div style={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #F8F9FA 0%, #E8D8C8 100%)',
            padding: '20px',
            fontFamily: "'Inter', sans-serif",
          }}>
            <header style={{
              textAlign: 'center',
              marginBottom: '40px',
            }}>
              <h1 style={{
                fontSize: '2.5rem',
                fontWeight: '700',
                color: '#2C3E50',
                margin: '0',
                fontFamily: 'Lora, serif',
              }}>
                🌿 OptoPomodoro
              </h1>
              <p style={{
                fontSize: '1.1rem',
                color: '#8B7D7B',
                margin: '8px 0 0 0',
              }}>
                Find your flow, achieve your goals
              </p>
            </header>

            <main>
              <TimerScreen />
            </main>
          </div>
        </ThemeProvider>
      </PersistGate>
    </Provider>
  );
}

export default App;
