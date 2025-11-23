import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { Session, SessionType, CreateSessionRequest, CompleteSessionRequest } from '../../types';

interface TimerState {
  currentSession: Session | null;
  isRunning: boolean;
  isPaused: boolean;
  remainingTime: number; // in seconds
  totalTime: number; // in seconds
  sessionType: SessionType;
  sessionsCompleted: number; // count in current work sequence
  workDuration: number; // in seconds
  shortBreakDuration: number; // in seconds
  longBreakDuration: number; // in seconds
  longBreakInterval: number; // after how many work sessions
  autoStartBreaks: boolean;
  autoStartWork: boolean;
  isLoading: boolean;
  error: string | null;
}

const initialState: TimerState = {
  currentSession: null,
  isRunning: false,
  isPaused: false,
  remainingTime: 25 * 60, // 25 minutes default
  totalTime: 25 * 60,
  sessionType: 'POMODORO',
  sessionsCompleted: 0,
  workDuration: 25 * 60,
  shortBreakDuration: 5 * 60,
  longBreakDuration: 15 * 60,
  longBreakInterval: 4,
  autoStartBreaks: false,
  autoStartWork: false,
  isLoading: false,
  error: null,
};

// Async thunks
export const createSession = createAsyncThunk<
  Session,
  CreateSessionRequest,
  { rejectValue: string }
>(
  'timer/createSession',
  async (sessionData, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: { token: string | null } };
      const token = state.auth.token;

      const response = await fetch('/api/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': token ? `Bearer ${token}` : '',
        },
        body: JSON.stringify(sessionData),
      });

      if (!response.ok) {
        let error = 'Failed to create session';
        try {
          const errorData = await response.json();
          error = errorData.message || error;
        } catch {
          // If response is not JSON, use status text
          error = response.statusText || `HTTP ${response.status}`;
        }
        throw new Error(error);
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Failed to create session');
    }
  }
);

export const startSession = createAsyncThunk<
  Session,
  void,
  { rejectValue: string }
>(
  'timer/startSession',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as {
        auth: { token: string | null };
        timer: TimerState;
      };
      const token = state.auth.token;
      const { currentSession } = state.timer;

      if (!currentSession) {
        throw new Error('No active session');
      }

      const response = await fetch(`/api/sessions/${currentSession.id}/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': token ? `Bearer ${token}` : '',
        },
      });

      if (!response.ok) {
        let error = 'Failed to start session';
        try {
          const errorData = await response.json();
          error = errorData.message || error;
        } catch {
          // If response is not JSON, use status text
          error = response.statusText || `HTTP ${response.status}`;
        }
        throw new Error(error);
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Failed to start session');
    }
  }
);

export const pauseSession = createAsyncThunk<
  Session,
  void,
  { rejectValue: string }
>(
  'timer/pauseSession',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as {
        auth: { token: string | null };
        timer: TimerState;
      };
      const token = state.auth.token;
      const { currentSession } = state.timer;

      if (!currentSession) {
        throw new Error('No active session');
      }

      const response = await fetch(`/api/sessions/${currentSession.id}/pause`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': token ? `Bearer ${token}` : '',
        },
      });

      if (!response.ok) {
        let error = 'Failed to pause session';
        try {
          const errorData = await response.json();
          error = errorData.message || error;
        } catch {
          // If response is not JSON, use status text
          error = response.statusText || `HTTP ${response.status}`;
        }
        throw new Error(error);
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Failed to pause session');
    }
  }
);

export const completeSession = createAsyncThunk<
  Session,
  CompleteSessionRequest,
  { rejectValue: string }
>(
  'timer/completeSession',
  async (completionData, { getState, rejectWithValue }) => {
    try {
      const state = getState() as {
        auth: { token: string | null };
        timer: TimerState;
      };
      const token = state.auth.token;
      const { currentSession } = state.timer;

      if (!currentSession) {
        throw new Error('No active session');
      }

      const response = await fetch(`/api/sessions/${currentSession.id}/complete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': token ? `Bearer ${token}` : '',
        },
        body: JSON.stringify(completionData),
      });

      if (!response.ok) {
        let error = 'Failed to complete session';
        try {
          const errorData = await response.json();
          error = errorData.message || error;
        } catch {
          // If response is not JSON, use status text
          error = response.statusText || `HTTP ${response.status}`;
        }
        throw new Error(error);
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Failed to complete session');
    }
  }
);

const timerSlice = createSlice({
  name: 'timer',
  initialState,
  reducers: {
    startTimer: (state) => {
      state.isRunning = true;
      state.isPaused = false;
    },
    pauseTimer: (state) => {
      state.isRunning = false;
      state.isPaused = true;
    },
    resetTimer: (state) => {
      state.isRunning = false;
      state.isPaused = false;
      state.remainingTime = state.totalTime;
    },
    skipSession: (state) => {
      state.isRunning = false;
      state.isPaused = false;

      // Determine next session type
      if (state.sessionType === 'POMODORO') {
        state.sessionsCompleted += 1;
        if (state.sessionsCompleted % state.longBreakInterval === 0) {
          state.sessionType = 'LONG_BREAK';
          state.totalTime = state.longBreakDuration;
        } else {
          state.sessionType = 'SHORT_BREAK';
          state.totalTime = state.shortBreakDuration;
        }
      } else {
        state.sessionType = 'POMODORO';
        state.totalTime = state.workDuration;
      }

      state.remainingTime = state.totalTime;
    },
    decrementTime: (state) => {
      if (state.remainingTime > 0) {
        state.remainingTime -= 1;
      }
    },
    setSessionType: (state, action: PayloadAction<SessionType>) => {
      state.sessionType = action.payload;
      switch (action.payload) {
        case 'POMODORO':
          state.totalTime = state.workDuration;
          break;
        case 'SHORT_BREAK':
          state.totalTime = state.shortBreakDuration;
          break;
        case 'LONG_BREAK':
          state.totalTime = state.longBreakDuration;
          break;
      }
      state.remainingTime = state.totalTime;
      state.isRunning = false;
      state.isPaused = false;
    },
    setDuration: (state, action: PayloadAction<{
      workDuration?: number;
      shortBreakDuration?: number;
      longBreakDuration?: number;
    }>) => {
      if (action.payload.workDuration !== undefined) {
        state.workDuration = action.payload.workDuration;
        if (state.sessionType === 'POMODORO') {
          state.totalTime = action.payload.workDuration;
          state.remainingTime = action.payload.workDuration;
        }
      }
      if (action.payload.shortBreakDuration !== undefined) {
        state.shortBreakDuration = action.payload.shortBreakDuration;
        if (state.sessionType === 'SHORT_BREAK') {
          state.totalTime = action.payload.shortBreakDuration;
          state.remainingTime = action.payload.shortBreakDuration;
        }
      }
      if (action.payload.longBreakDuration !== undefined) {
        state.longBreakDuration = action.payload.longBreakDuration;
        if (state.sessionType === 'LONG_BREAK') {
          state.totalTime = action.payload.longBreakDuration;
          state.remainingTime = action.payload.longBreakDuration;
        }
      }
    },
    setAutoStartSettings: (state, action: PayloadAction<{
      autoStartBreaks?: boolean;
      autoStartWork?: boolean;
    }>) => {
      if (action.payload.autoStartBreaks !== undefined) {
        state.autoStartBreaks = action.payload.autoStartBreaks;
      }
      if (action.payload.autoStartWork !== undefined) {
        state.autoStartWork = action.payload.autoStartWork;
      }
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Create Session
    builder
      .addCase(createSession.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(createSession.fulfilled, (state, action) => {
        state.isLoading = false;
        state.currentSession = action.payload;
        state.error = null;
      })
      .addCase(createSession.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to create session';
      });

    // Start Session
    builder
      .addCase(startSession.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(startSession.fulfilled, (state, action) => {
        state.isLoading = false;
        state.currentSession = action.payload;
        state.isRunning = true;
        state.isPaused = false;
      })
      .addCase(startSession.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to start session';
      });

    // Pause Session
    builder
      .addCase(pauseSession.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(pauseSession.fulfilled, (state, action) => {
        state.isLoading = false;
        state.currentSession = action.payload;
        state.isRunning = false;
        state.isPaused = true;
      })
      .addCase(pauseSession.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to pause session';
      });

    // Complete Session
    builder
      .addCase(completeSession.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(completeSession.fulfilled, (state, action) => {
        state.isLoading = false;
        const completedSession = action.payload;

        // Update the session and determine next session type
        state.currentSession = null;
        state.isRunning = false;
        state.isPaused = false;

        if (completedSession.type === 'POMODORO') {
          state.sessionsCompleted += 1;
        }

        // Auto-start next session if enabled
        if (completedSession.type === 'POMODORO' && state.autoStartBreaks) {
          // Start break automatically
          if (state.sessionsCompleted % state.longBreakInterval === 0) {
            state.sessionType = 'LONG_BREAK';
            state.totalTime = state.longBreakDuration;
          } else {
            state.sessionType = 'SHORT_BREAK';
            state.totalTime = state.shortBreakDuration;
          }
          state.remainingTime = state.totalTime;
          state.isRunning = true;
          state.isPaused = false;
        } else if (completedSession.type !== 'POMODORO' && state.autoStartWork) {
          // Start work automatically
          state.sessionType = 'POMODORO';
          state.totalTime = state.workDuration;
          state.remainingTime = state.totalTime;
          state.isRunning = true;
          state.isPaused = false;
        } else {
          // Don't auto-start, reset to work session
          state.sessionType = 'POMODORO';
          state.totalTime = state.workDuration;
          state.remainingTime = state.totalTime;
        }

        state.error = null;
      })
      .addCase(completeSession.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to complete session';
      });
  },
});

export const {
  startTimer,
  pauseTimer,
  resetTimer,
  skipSession,
  decrementTime,
  setSessionType,
  setDuration,
  setAutoStartSettings,
  clearError,
} = timerSlice.actions;

// Selectors
export const timerSelectors = {
  selectCurrentSession: (state: { timer: TimerState }) => state.timer.currentSession,
  selectTimerState: (state: { timer: TimerState }) => state.timer,
  selectIsRunning: (state: { timer: TimerState }) => state.timer.isRunning,
  selectIsPaused: (state: { timer: TimerState }) => state.timer.isPaused,
  selectRemainingTime: (state: { timer: TimerState }) => state.timer.remainingTime,
  selectTotalTime: (state: { timer: TimerState }) => state.timer.totalTime,
  selectSessionType: (state: { timer: TimerState }) => state.timer.sessionType,
  selectSessionsCompleted: (state: { timer: TimerState }) => state.timer.sessionsCompleted,
  selectWorkDuration: (state: { timer: TimerState }) => state.timer.workDuration,
  selectShortBreakDuration: (state: { timer: TimerState }) => state.timer.shortBreakDuration,
  selectLongBreakDuration: (state: { timer: TimerState }) => state.timer.longBreakDuration,
  selectAutoStartBreaks: (state: { timer: TimerState }) => state.timer.autoStartBreaks,
  selectAutoStartWork: (state: { timer: TimerState }) => state.timer.autoStartWork,
};

export { timerSlice };
export type { TimerState };