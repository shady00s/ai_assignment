import { useSelector, useDispatch } from 'react-redux';
import { useCallback, useEffect } from 'react';
import { selectTimerState, selectIsRunning, selectIsPaused, selectRemainingTime, selectTotalTime, selectSessionType, selectSessionsCompleted, startTimer, pauseTimer, skipSession, setSessionType } from '@/store';
import { useCreateSessionMutation, useCompleteSessionMutation } from '@/store/api';
 

interface UseTimerLogicOptions {
  currentTaskId?: string;
  onSessionComplete?: (sessionId: string) => void;
}

interface UseTimerLogicReturn {
  isRunning: boolean;
  isPaused: boolean;
  remainingTime: number;
  totalTime: number;
  sessionType: 'POMODORO' | 'SHORT_BREAK' | 'LONG_BREAK';
  sessionsCompleted: number;
  progress: number;
  canStart: boolean;
  canPause: boolean;
  canSkip: boolean;
  canComplete: boolean;
  handleStart: () => void;
  handlePause: () => void;
  handleResume: () => void;
  handleSkip: () => void;
  handleComplete: () => void;
  handleSessionTypeChange: (type: 'POMODORO' | 'SHORT_BREAK' | 'LONG_BREAK') => void;
  formatTime: (seconds: number) => string;
  isLoading: boolean;
}

export const useTimerLogic = (options: UseTimerLogicOptions = {}): UseTimerLogicReturn => {
  const dispatch = useDispatch();
  const timerState = useSelector(selectTimerState);
  const isRunning = useSelector(selectIsRunning);
  const isPaused = useSelector(selectIsPaused);
  const remainingTime = useSelector(selectRemainingTime);
  const totalTime = useSelector(selectTotalTime);
  const sessionType = useSelector(selectSessionType);
  const sessionsCompleted = useSelector(selectSessionsCompleted);

  const [createSession, { isLoading: isCreatingSession }] = useCreateSessionMutation();
  const [completeSession, { isLoading: isCompletingSession }] = useCompleteSessionMutation();

  const isLoading = isCreatingSession || isCompletingSession;
  const progress = totalTime > 0 ? (totalTime - remainingTime) / totalTime : 0;

  // Calculate control permissions
  const canStart = !isRunning && !isPaused && totalTime > 0;
  const canPause = isRunning;
  const canSkip = isRunning || isPaused;
  const canComplete = isRunning || isPaused;

  const handleStart = useCallback(async () => {
    try {
      // Create a new session if needed
      if (!timerState.currentSession) {
        await createSession({
          type: sessionType,
          taskId: options.currentTaskId,
          plannedDuration: Math.floor(totalTime / 60), // Convert to minutes
        }).unwrap();
      }

      dispatch(startTimer());
    } catch (error) {
      console.error('Failed to start session:', error);
    }
  }, [dispatch, sessionType, timerState.currentSession, options.currentTaskId, totalTime, createSession]);

  const handlePause = useCallback(() => {
    dispatch(pauseTimer());
  }, [dispatch]);

  const handleResume = useCallback(() => {
    dispatch(startTimer());
  }, [dispatch]);

  const handleSkip = useCallback(() => {
    dispatch(skipSession());
  }, [dispatch]);

  const handleComplete = useCallback(async () => {
    try {
      if (timerState.currentSession) {
        await completeSession({
          id: timerState.currentSession.id,
          quality: 5, // Default quality score
          notes: '',
          interruptions: 0,
        }).unwrap();

        options.onSessionComplete?.(timerState.currentSession.id);
      }

      dispatch(completeSession());
    } catch (error) {
      console.error('Failed to complete session:', error);
    }
  }, [dispatch, timerState.currentSession, completeSession, options.onSessionComplete]);

  const handleSessionTypeChange = useCallback((type: 'POMODORO' | 'SHORT_BREAK' | 'LONG_BREAK') => {
    if (!isRunning) {
      dispatch(setSessionType(type));
    }
  }, [dispatch, isRunning]);

  const formatTime = useCallback((seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }, []);

  // Auto-complete when timer reaches zero
  useEffect(() => {
    if (isRunning && remainingTime === 0) {
      handleComplete();
    }
  }, [isRunning, remainingTime, handleComplete]);

  return {
    isRunning,
    isPaused,
    remainingTime,
    totalTime,
    sessionType,
    sessionsCompleted,
    progress,
    canStart,
    canPause,
    canSkip,
    canComplete,
    handleStart,
    handlePause,
    handleResume,
    handleSkip,
    handleComplete,
    handleSessionTypeChange,
    formatTime,
    isLoading,
  };
};