import { configureStore } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import { combineReducers } from '@reduxjs/toolkit';

// Import slices
import { authSlice, authSelectors } from './slices/authSlice';
import { timerSlice, timerSelectors } from './slices/timerSlice';
import { tasksSlice, tasksSelectors } from './slices/tasksSlice';
import { uiSlice, uiSelectors } from './slices/uiSlice';
import { apiSlice } from './api/apiSlice';
import { wellnessApi } from './api/wellnessApi';

// Import timer actions
import {
  createSession as timerCreateSession,
  startSession as timerStartSession,
  pauseSession as timerPauseSession,
  completeSession as timerCompleteSession,
  setSessionType as timerSetSessionType,
  skipSession as timerSkipSession
} from './slices/timerSlice';

const persistConfig = {
  key: 'root',
  storage,
  whitelist: ['auth', 'timer', 'tasks'], // Only persist these slices
  blacklist: ['api'], // Don't persist API cache
};

const rootReducer = combineReducers({
  auth: authSlice.reducer,
  timer: timerSlice.reducer,
  tasks: tasksSlice.reducer,
  ui: uiSlice.reducer,
  api: apiSlice.reducer,
  wellnessApi: wellnessApi.reducer,
});

const persistedReducer = persistReducer(persistConfig, rootReducer);

export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE', 'persist/REGISTER'],
        ignoredPaths: ['api', 'wellnessApi'],
      },
    }).concat(apiSlice.middleware, wellnessApi.middleware),
  devTools: process.env.NODE_ENV !== 'production',
});

export const persistor = persistStore(store);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Export timer actions
export const createSession = timerCreateSession;
export const startSession = timerStartSession;
export const pauseSession = timerPauseSession;
export const completeSession = timerCompleteSession;
export const setSessionType = timerSetSessionType;
export const skipSession = timerSkipSession;

// Export timer slice actions
export const {
  startTimer,
  pauseTimer,
  resetTimer,
  decrementTime,
  setDuration,
  setAutoStartSettings,
  clearError,
} = timerSlice.actions;

// Export selectors
export const {
  selectUser,
  selectIsAuthenticated,
  selectToken
} = authSelectors;

export const {
  selectCurrentSession,
  selectTimerState,
  selectIsRunning,
  selectIsPaused,
  selectRemainingTime,
  selectTotalTime,
  selectSessionType,
  selectSessionsCompleted,
  selectWorkDuration,
  selectShortBreakDuration,
  selectLongBreakDuration,
} = timerSelectors;

export const {
  selectAllTasks,
  selectTasksByStatus,
  selectTasksByPriority,
  selectCurrentTask,
  selectTaskById,
} = tasksSelectors;

export const {
  selectSidebarOpen,
  selectTheme,
  selectCurrentView,
  selectLoading,
  selectError,
  selectNotifications,
} = uiSelectors;