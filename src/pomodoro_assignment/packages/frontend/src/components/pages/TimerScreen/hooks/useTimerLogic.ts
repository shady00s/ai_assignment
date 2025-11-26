import { useSelector, useDispatch } from 'react-redux';
import { useCallback, useEffect, useRef } from 'react';
import { selectTimerState, selectIsRunning, selectIsPaused, selectRemainingTime, selectTotalTime, selectSessionType, selectSessionsCompleted, startTimer, pauseTimer, skipSession, setSessionType, setCurrentSession } from '@/store';
import { SessionType } from '@/types';
import {
  useCreateSessionMutation,
  useCompleteSessionMutation,
   useStartSessionMutation,
  usePauseSessionMutation,

} from '@/store/api';
import { useGetActiveSessionQuery, useSkipSessionMutation } from '@/store/api/apiSlice';
 

interface UseTimerLogicOptions {
  currentTaskId?: string;
  onSessionComplete?: (sessionId: string) => void;
}

interface UseTimerLogicReturn {
  isRunning: boolean;
  isPaused: boolean;
  remainingTime: number;
  totalTime: number;
  sessionType: SessionType;
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
  handleSessionTypeChange: (type: SessionType) => void;
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
  const [skipSessionMutation, { isLoading: isSkippingSession }] = useSkipSessionMutation();
  const [startSession, { isLoading: isStartingSession }] = useStartSessionMutation();
  const [pauseSession, { isLoading: isPausingSession }] = usePauseSessionMutation();

  // Sync with backend active session
  const { data: activeSession, refetch: refetchActiveSession } = useGetActiveSessionQuery();

  const isLoading = isCreatingSession || isCompletingSession || isSkippingSession || isStartingSession || isPausingSession;
  const progress = totalTime > 0 ? (totalTime - remainingTime) / totalTime : 0;

  // Calculate control permissions
  const canStart = !isRunning && !isPaused && totalTime > 0;
  const canPause = isRunning;
  const canSkip = isRunning || isPaused;
  const canComplete = isRunning || isPaused;

  const handleStart = useCallback(async () => {
    try {
      let currentSession = timerState.currentSession || activeSession;

      // Create a new session if needed
      if (!currentSession) {
        const duration = Math.floor(totalTime / 60); // Convert to minutes
        if (duration < 1 || duration > 180) {
          throw new Error(`Session duration must be between 1 and 180 minutes, got ${duration}`);
        }

        const createdSession = await createSession({
          type: sessionType,
          taskId: options.currentTaskId,
          duration, // Backend expects 'duration', not 'plannedDuration'
        }).unwrap();
        currentSession = createdSession;

        // Update local timer state with the created session
        dispatch(setCurrentSession(createdSession));
      }

      // Start the session in backend
      if (currentSession?.id) {
        try {
          await startSession(currentSession.id).unwrap();
        } catch (startError: any) {
          // If session is already started, that's actually fine
          if (startError?.status === 403 && (startError?.data?.error === 'Forbidden' || startError?.data?.message === 'Session already started')) {
            console.log('Session already started, proceeding with timer');
          } else {
            console.error('Failed to start session:', startError);
            throw startError;
          }
        }
      }

      dispatch(startTimer());
    } catch (error) {
      console.error('Failed to start session:', error);
    }
  }, [dispatch, sessionType, timerState.currentSession, activeSession, options.currentTaskId, totalTime, createSession, startSession]);

  const handlePause = useCallback(async () => {
    try {
      // Pause the session in backend if it exists
      if (timerState.currentSession?.id) {
        await pauseSession(timerState.currentSession.id).unwrap();
      }
      dispatch(pauseTimer());
    } catch (error) {
      console.error('Failed to pause session:', error);
      // Still pause locally even if backend fails
      dispatch(pauseTimer());
    }
  }, [dispatch, timerState.currentSession, pauseSession]);

  const handleResume = useCallback(async () => {
    try {
      // Resume the session in backend if it was paused
      if (timerState.currentSession?.id) {
        await startSession(timerState.currentSession.id).unwrap();
      }
      dispatch(startTimer());
    } catch (error) {
      console.error('Failed to resume session:', error);
      dispatch(startTimer());
    }
  }, [dispatch, timerState.currentSession, startSession]);

  const handleSkip = useCallback(async () => {
    try {
      // If there's an active session, skip it using the dedicated skip endpoint
      if (timerState.currentSession?.id) {
        await skipSessionMutation({
          id: timerState.currentSession.id,
          notes: 'Session skipped',
        }).unwrap();

        // Notify callback if provided
        options.onSessionComplete?.(timerState.currentSession.id);
      }

      // Clear current session from Redux state
      dispatch(skipSession());
    } catch (error) {
      console.error('Failed to skip session:', error);
      // Still skip locally even if backend fails
      dispatch(skipSession());
    }
  }, [dispatch, timerState.currentSession, skipSessionMutation, options]);

  const handleComplete = useCallback(async () => {
    try {
      if (timerState.currentSession?.id) {
        // Complete session in backend using RTK Query mutation
        await completeSession({
          id: timerState.currentSession.id,
          quality: 5, // Default quality score
          notes: 'Session completed successfully',
        }).unwrap();

        // Notify callback if provided
        options.onSessionComplete?.(timerState.currentSession.id);

        // Clear local timer state (the backend session is now completed)
        dispatch(setCurrentSession(null)); // Clear the current session
        dispatch(skipSession()); // Advance to next session type
      }
    } catch (error) {
      console.error('Failed to complete session:', error);
      // Still clear local state even if backend fails
      dispatch(setCurrentSession(null));
      dispatch(skipSession());
    }
  }, [dispatch, timerState.currentSession, completeSession, options]);

  const handleSessionTypeChange = useCallback((type: SessionType) => {
    if (!isRunning) {
      dispatch(setSessionType(type));
    }
  }, [dispatch, isRunning]);

  const formatTime = useCallback((seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }, []);

  // Sync timer state with backend active session
  useEffect(() => {
    if (activeSession && activeSession !== timerState.currentSession) {
      // Update local state with backend session data
      // This could dispatch actions to sync timer state if needed
    }
  }, [activeSession, timerState.currentSession]);

  // Auto-complete when timer reaches zero
  const handleCompleteRef = useRef(handleComplete);
  handleCompleteRef.current = handleComplete;

  useEffect(() => {
    if (isRunning && remainingTime === 0) {
      handleCompleteRef.current();
    }
  }, [isRunning, remainingTime]);

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