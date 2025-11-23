# OptoPomodoro Frontend Implementation Plan
**React 18 + TypeScript + Vite PWA**

## Team Overview
**Target Framework**: React 18.3.1 with TypeScript 5.5.x
**Build System**: Vite 5.4.x with PWA plugin
**State Management**: Redux Toolkit 2.2.x + RTK Query 2.2.x
**Styling**: Styled-components 6.1.x with Design System
**Animation**: Framer Motion 11.x
**Testing**: Vitest 2.x + React Testing Library 16.x + Cypress 13.x

## Phase 1: Foundation Setup (Week 1-2)

### 1.1 Package Installation & Configuration

#### Core Dependencies
```bash
# Core framework
pnpm add react@18.3.1 react-dom@18.3.1 typescript@5.5.4
pnpm add -D @types/react@18.3.3 @types/react-dom@18.3.0

# Build system & PWA
pnpm add -D vite@5.4.2 @vitejs/plugin-react@4.3.1
pnpm add -D vite-plugin-pwa@0.20.1 workbox-window@7.1.0

# State management
pnpm add @reduxjs/toolkit@2.2.5 react-redux@9.1.2
pnpm add @reduxjs/react-persist@2.1.0

# Routing
pnpm add react-router-dom@6.26.1

# Styling & Design
pnpm add styled-components@6.1.11
pnpm add -D @types/styled-components@5.1.34

# Animations
pnpm add framer-motion@11.3.19

# PWA & Offline
pnpm add idb@8.0.0

# Development tools
pnpm add -D eslint@9.9.1 @typescript-eslint/eslint-plugin@8.4.0
pnpm add -D prettier@3.3.3 husky@9.1.5 lint-staged@15.2.9

# Testing
pnpm add -D vitest@2.0.5 @vitest/ui@2.0.5
pnpm add -D @testing-library/react@16.0.0 @testing-library/jest-dom@6.5.0
pnpm add -D @testing-library/user-event@14.5.2
pnpm add -D cypress@13.13.3 @cypress/react@6.0.0
```

#### Project Structure
```
packages/frontend/
├── public/
│   ├── icons/
│   │   ├── pwa-192x192.png
│   │   └── pwa-512x512.png
│   ├── manifest.json
│   └── sw.js
├── src/
│   ├── components/
│   │   ├── atoms/
│   │   ├── molecules/
│   │   ├── organisms/
│   │   ├── templates/
│   │   └── pages/
│   ├── hooks/
│   ├── store/
│   │   ├── slices/
│   │   ├── api/
│   │   └── index.ts
│   ├── services/
│   │   ├── api.ts
│   │   ├── offlineService.ts
│   │   └── notificationService.ts
│   ├── utils/
│   ├── themes/
│   │   ├── ZenTheme.ts
│   │   └── designTokens.ts
│   ├── types/
│   ├── assets/
│   │   ├── sounds/
│   │   ├── images/
│   │   └── icons/
│   ├── App.tsx
│   ├── main.tsx
│   └── vite-env.d.ts
├── cypress/
├── tests/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── vitest.config.ts
└── tailwind.config.js
```

### 1.2 Vite Configuration

#### vite.config.ts
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'masked-icon.svg'],
      manifest: {
        name: 'OptoPomodoro',
        short_name: 'OptoPomodoro',
        description: 'Zen-inspired productivity for Optomatica teams',
        theme_color: '#7A8B7F',
        background_color: '#F4E4D4',
        display: 'standalone',
        orientation: 'portrait',
        scope: '/',
        start_url: '/',
        icons: [
          {
            src: 'icons/pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png',
          },
          {
            src: 'icons/pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
          },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\.optopomodoro\.com/,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 60 * 60 * 24,
              },
            },
          },
        ],
      },
    }),
  ],
  resolve: {
    alias: {
      '@': '/src',
      '@components': '/src/components',
      '@hooks': '/src/hooks',
      '@store': '/src/store',
      '@utils': '/src/utils',
      '@themes': '/src/themes',
      '@assets': '/src/assets',
      '@services': '/src/services',
      '@types': '/src/types',
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
        secure: false,
      },
      '/socket.io': {
        target: 'http://localhost:3001',
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
```

### 1.3 Design System Implementation

#### Design Tokens (themes/designTokens.ts)
```typescript
export const designTokens = {
  colors: {
    primary: {
      main: '#7A8B7F',      // Moss Green
      light: '#9AA895',
      dark: '#5F6E63',
    },
    secondary: {
      main: '#6B8E9F',      // Water Blue
      light: '#8DA4B1',
      dark: '#537281',
    },
    accent: {
      main: '#E67E50',      // Sunrise Orange
      light: '#EB9D7A',
      dark: '#C46441',
    },
    neutral: {
      50: '#F4E4D4',        // Warm Sand
      100: '#E8D8C8',
      200: '#D4C5B9',      // Zen Stone
      300: '#C0B2A5',
      400: '#8B7D7B',      // Stone Gray
      500: '#2C3E50',      // Charcoal
    },
    success: '#7FA870',     // Sage Green
    warning: '#F4A261',     // Warm Amber
    error: '#C85A5A',       // Soft Red
    info: '#6B8E9F',        // Sky Blue
  },

  typography: {
    fontFamily: {
      primary: 'Inter, sans-serif',
      secondary: 'Lora, serif',
    },
    fontSize: {
      xs: '12px',
      sm: '14px',
      base: '16px',
      lg: '18px',
      xl: '24px',
      '2xl': '32px',
      '3xl': '48px',
    },
    fontWeight: {
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.75,
    },
  },

  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '32px',
    xl: '64px',
  },

  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '16px',
    full: '50%',
  },

  shadows: {
    sm: '0 1px 3px rgba(0, 0, 0, 0.1)',
    md: '0 4px 6px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 25px rgba(0, 0, 0, 0.1)',
  },

  animation: {
    duration: {
      fast: '150ms',
      normal: '300ms',
      slow: '500ms',
    },
    easing: {
      easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    },
  },
};
```

#### Zen Theme (themes/ZenTheme.ts)
```typescript
import { DefaultTheme } from 'styled-components';
import { designTokens } from './designTokens';

export const ZenTheme: DefaultTheme = {
  ...designTokens,
  breakpoints: {
    mobile: '320px',
    tablet: '768px',
    desktop: '1024px',
  },
  components: {
    Button: {
      primary: {
        backgroundColor: designTokens.colors.primary.main,
        color: '#FFFFFF',
        borderRadius: designTokens.borderRadius.md,
        padding: `${designTokens.spacing.sm} ${designTokens.spacing.md}`,
        border: 'none',
        cursor: 'pointer',
        transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.easeInOut}`,

        '&:hover': {
          backgroundColor: designTokens.colors.primary.dark,
        },

        '&:disabled': {
          opacity: 0.6,
          cursor: 'not-allowed',
        },
      },
      secondary: {
        backgroundColor: 'transparent',
        color: designTokens.colors.primary.main,
        border: `2px solid ${designTokens.colors.primary.main}`,
        borderRadius: designTokens.borderRadius.md,
        padding: `${designTokens.spacing.sm} ${designTokens.spacing.md}`,
        cursor: 'pointer',
        transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.easeInOut}`,

        '&:hover': {
          backgroundColor: designTokens.colors.primary.main,
          color: '#FFFFFF',
        },
      },
    },
    Card: {
      backgroundColor: '#FFFFFF',
      borderRadius: designTokens.borderRadius.lg,
      boxShadow: designTokens.shadows.md,
      padding: designTokens.spacing.lg,
      transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.easeOut}`,

      '&:hover': {
        boxShadow: designTokens.shadows.lg,
        transform: 'translateY(-2px)',
      },
    },
    Timer: {
      fontSize: designTokens.fontSize['3xl'],
      fontWeight: designTokens.fontWeight.bold,
      color: designTokens.colors.neutral[500],
      fontFamily: designTokens.typography.fontFamily.secondary,
    },
  },
};
```

### 1.4 Redux Store Setup

#### Store Configuration (store/index.ts)
```typescript
import { configureStore } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import { combineReducers } from '@reduxjs/toolkit';

import { authSlice } from './slices/authSlice';
import { timerSlice } from './slices/timerSlice';
import { tasksSlice } from './slices/tasksSlice';
import { uiSlice } from './slices/uiSlice';
import { apiSlice } from './api/apiSlice';

const persistConfig = {
  key: 'root',
  storage,
  whitelist: ['auth', 'timer', 'tasks'], // Only persist these slices
};

const rootReducer = combineReducers({
  auth: authSlice.reducer,
  timer: timerSlice.reducer,
  tasks: tasksSlice.reducer,
  ui: uiSlice.reducer,
  api: apiSlice.reducer,
});

const persistedReducer = persistReducer(persistConfig, rootReducer);

export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }).concat(apiSlice.middleware),
  devTools: process.env.NODE_ENV !== 'production',
});

export const persistor = persistStore(store);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

#### API Slice (store/api/apiSlice.ts)
```typescript
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import type { RootState } from '../index';

export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token;
      if (token) {
        headers.set('authorization', `Bearer ${token}`);
      }
      return headers;
    },
  }),
  tagTypes: ['User', 'Task', 'Session', 'Team', 'Achievement'],
  endpoints: (builder) => ({
    // Auth endpoints
    login: builder.mutation<LoginResponse, LoginRequest>({
      query: (credentials) => ({
        url: 'auth/login',
        method: 'POST',
        body: credentials,
      }),
    }),

    register: builder.mutation<RegisterResponse, RegisterRequest>({
      query: (userData) => ({
        url: 'auth/register',
        method: 'POST',
        body: userData,
      }),
    }),

    // User endpoints
    getProfile: builder.query<User, void>({
      query: () => 'users/profile',
      providesTags: ['User'],
    }),

    updateProfile: builder.mutation<User, Partial<User>>({
      query: (userData) => ({
        url: 'users/profile',
        method: 'PUT',
        body: userData,
      }),
      invalidatesTags: ['User'],
    }),

    // Task endpoints
    getTasks: builder.query<Task[], { status?: TaskStatus; teamId?: string }>({
      query: (params) => ({
        url: 'tasks',
        params,
      }),
      providesTags: ['Task'],
    }),

    createTask: builder.mutation<Task, CreateTaskRequest>({
      query: (taskData) => ({
        url: 'tasks',
        method: 'POST',
        body: taskData,
      }),
      invalidatesTags: ['Task'],
    }),

    updateTask: builder.mutation<Task, { id: string; updates: Partial<Task> }>({
      query: ({ id, updates }) => ({
        url: `tasks/${id}`,
        method: 'PUT',
        body: updates,
      }),
      invalidatesTags: ['Task'],
    }),

    // Session endpoints
    getSessions: builder.query<Session[], { startDate?: string; endDate?: string }>({
      query: (params) => ({
        url: 'sessions',
        params,
      }),
      providesTags: ['Session'],
    }),

    createSession: builder.mutation<Session, CreateSessionRequest>({
      query: (sessionData) => ({
        url: 'sessions',
        method: 'POST',
        body: sessionData,
      }),
      invalidatesTags: ['Session'],
    }),

    completeSession: builder.mutation<Session, { id: string; quality?: number }>({
      query: ({ id, quality }) => ({
        url: `sessions/${id}/complete`,
        method: 'POST',
        body: { quality },
      }),
      invalidatesTags: ['Session'],
    }),
  }),
});

export const {
  useLoginMutation,
  useRegisterMutation,
  useGetProfileQuery,
  useUpdateProfileMutation,
  useGetTasksQuery,
  useCreateTaskMutation,
  useUpdateTaskMutation,
  useGetSessionsQuery,
  useCreateSessionMutation,
  useCompleteSessionMutation,
} = apiSlice;
```

## Phase 2: Core Components Implementation (Week 3-6)

### 2.1 Atomic Components

#### Button Component
```typescript
// components/atoms/Button/Button.tsx
import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  children: React.ReactNode;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
  className?: string;
}

const StyledButton = styled(motion.button)<{
  $variant: 'primary' | 'secondary' | 'ghost';
  $size: 'small' | 'medium' | 'large';
}>`
  ${({ theme, $variant, $size }) => theme.components.Button[$variant]};
  font-size: ${({ theme, $size }) => {
    switch ($size) {
      case 'small': return theme.typography.fontSize.sm;
      case 'medium': return theme.typography.fontSize.base;
      case 'large': return theme.typography.fontSize.lg;
      default: return theme.typography.fontSize.base;
    }
  }};
  padding: ${({ theme, $size }) => {
    switch ($size) {
      case 'small': return `${theme.spacing.xs} ${theme.spacing.sm}`;
      case 'medium': return `${theme.spacing.sm} ${theme.spacing.md}`;
      case 'large': return `${theme.spacing.md} ${theme.spacing.lg}`;
      default: return `${theme.spacing.sm} ${theme.spacing.md}`;
    }
  }};
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  text-decoration: none;
  position: relative;
  overflow: hidden;

  &:focus {
    outline: 2px solid ${({ theme }) => theme.colors.primary.main};
    outline-offset: 2px;
  }

  &:disabled {
    cursor: not-allowed;
    opacity: 0.6;
  }
`;

const LoadingSpinner = styled.div`
  width: 16px;
  height: 16px;
  border: 2px solid transparent;
  border-top: 2px solid currentColor;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  icon,
  children,
  onClick,
  type = 'button',
  className,
}) => {
  return (
    <StyledButton
      as={motion.button}
      $variant={variant}
      $size={size}
      disabled={disabled || loading}
      onClick={onClick}
      type={type}
      className={className}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      transition={{ duration: 0.15 }}
    >
      {loading && <LoadingSpinner />}
      {!loading && icon}
      {children}
    </StyledButton>
  );
};
```

### 2.2 Timer Components

#### Circular Progress Timer
```typescript
// components/organisms/CircularTimer/CircularTimer.tsx
import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useAppSelector, useAppDispatch } from '@/hooks/redux';
import {
  startSession,
  pauseSession,
  completeSession,
  selectCurrentSession,
  selectTimerState
} from '@/store/slices/timerSlice';

interface CircularTimerProps {
  size?: number;
  strokeWidth?: number;
  showControls?: boolean;
  className?: string;
}

const TimerContainer = styled.div<{ $size: number }>`
  width: ${({ $size }) => $size}px;
  height: ${({ $size }) => $size}px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const SVGContainer = styled.svg`
  transform: rotate(-90deg);
  filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
`;

const ProgressCircle = styled.circle<{ $progress: number }>`
  stroke-dasharray: ${({ $circumference }) => $circumference};
  stroke-dashoffset: ${({ $circumference, $progress }) => $circumference * (1 - $progress)};
  transition: stroke-dashoffset 1s linear;
  stroke-linecap: round;
`;

const TimerText = styled.div<{ $size: number }>`
  position: absolute;
  font-size: ${({ $size }) => Math.floor($size / 8)}px;
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};
  text-align: center;
  line-height: 1.2;
`;

const ControlsContainer = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.md};
  margin-top: ${({ theme }) => theme.spacing.lg};
`;

const ZenGarden = styled.div`
  position: absolute;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0.1;
  font-size: 2rem;
`;

export const CircularTimer: React.FC<CircularTimerProps> = ({
  size = 240,
  strokeWidth = 8,
  showControls = true,
  className,
}) => {
  const dispatch = useAppDispatch();
  const currentSession = useAppSelector(selectCurrentSession);
  const { isRunning, isPaused, remainingTime, totalTime } = useAppSelector(selectTimerState);

  const [progress, setProgress] = useState(0);

  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;

  useEffect(() => {
    if (totalTime > 0) {
      setProgress((totalTime - remainingTime) / totalTime);
    } else {
      setProgress(0);
    }
  }, [remainingTime, totalTime]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleStart = () => {
    if (currentSession) {
      dispatch(startSession());
    }
  };

  const handlePause = () => {
    dispatch(pauseSession());
  };

  const handleComplete = () => {
    if (currentSession) {
      dispatch(completeSession({ quality: 5 }));
    }
  };

  return (
    <div className={className}>
      <TimerContainer $size={size}>
        <ZenGarden>
          {isRunning ? '🌿' : isPaused ? '⏸️' : '🪨'}
        </ZenGarden>

        <SVGContainer width={size} height={size}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#F4E4D4"
            strokeWidth={strokeWidth}
            fill="none"
          />
          <ProgressCircle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={isRunning ? '#7A8B7F' : '#6B8E9F'}
            strokeWidth={strokeWidth}
            fill="none"
            $circumference={circumference}
            $progress={progress}
          />
        </SVGContainer>

        <TimerText $size={size}>
          <div>{formatTime(remainingTime)}</div>
          {currentSession && (
            <div style={{ fontSize: '0.4em', opacity: 0.7 }}>
              {currentSession.type === 'POMODORO' ? 'Focus Time' : 'Break'}
            </div>
          )}
        </TimerText>
      </TimerContainer>

      {showControls && (
        <ControlsContainer>
          {!isRunning ? (
            <Button
              variant="primary"
              size="large"
              icon="▶️"
              onClick={handleStart}
              disabled={!currentSession}
            >
              Start
            </Button>
          ) : (
            <Button
              variant="secondary"
              size="large"
              icon="⏸️"
              onClick={handlePause}
            >
              Pause
            </Button>
          )}

          <Button
            variant="ghost"
            size="large"
            icon="⏹️"
            onClick={handleComplete}
            disabled={!isRunning && !isPaused}
          >
            Complete
          </Button>
        </ControlsContainer>
      )}
    </div>
  );
};
```

## Phase 3: Screen Implementation (Week 3-6)

### 3.1 Timer Screen
```typescript
// components/pages/TimerPage/TimerPage.tsx
import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { CircularTimer } from '@/components/organisms/CircularTimer';
import { TaskSelector } from '@/components/molecules/TaskSelector';
import { SessionStatus } from '@/components/molecules/SessionStatus';
import { ZenGarden } from '@/components/molecules/ZenGarden';

const TimerPageContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing.lg};
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.neutral[50]} 0%, ${({ theme }) => theme.colors.neutral[100]} 100%);
`;

const MainContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xl};
  max-width: 600px;
  width: 100%;
`;

const ControlsSection = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.lg};
  width: 100%;
`;

export const TimerPage: React.FC = () => {
  return (
    <TimerPageContainer>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 style={{
          fontSize: '2rem',
          color: '#2C3E50',
          marginBottom: '2rem',
          textAlign: 'center',
          fontFamily: 'Lora, serif'
        }}>
          OptoPomodoro
        </h1>
      </motion.div>

      <MainContent>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <ZenGarden />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <CircularTimer size={280} strokeWidth={10} />
        </motion.div>

        <ControlsSection>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            style={{ width: '100%' }}
          >
            <SessionStatus />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            style={{ width: '100%' }}
          >
            <TaskSelector />
          </motion.div>
        </ControlsSection>
      </MainContent>
    </TimerPageContainer>
  );
};
```

This implementation plan provides:

1. **Complete foundation setup** with all compatible package versions
2. **Comprehensive design system** following the Zen theme requirements
3. **Robust state management** with Redux Toolkit and persistence
4. **Modular component architecture** using Atomic Design principles
5. **PWA capabilities** with offline functionality
6. **Type-safe API integration** with RTK Query
7. **Smooth animations** with Framer Motion
8. **Responsive design** system supporting all device sizes

The plan ensures the frontend team has everything needed to build a professional, performant, and user-friendly Pomodoro application that meets all the specified requirements from the UI/UX documentation.