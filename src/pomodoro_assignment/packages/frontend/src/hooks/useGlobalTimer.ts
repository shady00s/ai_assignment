import { useEffect, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store';
import { decrementTime } from '../store/slices/timerSlice';

/**
 * Global timer hook that manages the timer interval regardless of component mounting state.
 * This ensures the timer continues running even when navigating between screens.
 */
export const useGlobalTimer = () => {
  const dispatch = useDispatch();
  const isRunning = useSelector((state: RootState) => state.timer.isRunning);
  const remainingTime = useSelector((state: RootState) => state.timer.remainingTime);

  // Use a ref to store the interval reference that persists across re-renders
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
   const lastTimeRef = useRef<number>(0);

  useEffect(() => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Start new interval if timer is running
    if (isRunning && remainingTime > 0) {
      intervalRef.current = setInterval(() => {
        dispatch(decrementTime());
      }, 1000);
      lastTimeRef.current = Date.now();
    }

    // Cleanup function
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isRunning, remainingTime, dispatch]);

  // Handle visibility change (tab switching, app backgrounding)
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Page is hidden, record current time
        lastTimeRef.current = Date.now();
      } else {
        // Page is visible again, check if timer was running
        if (isRunning && remainingTime > 0) {
          const timePassed = Math.floor((Date.now() - lastTimeRef.current) / 1000);
          if (timePassed > 0) {
            // Update timer for the time that passed while page was hidden
            for (let i = 0; i < Math.min(timePassed, remainingTime); i++) {
              dispatch(decrementTime());
            }
          }
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isRunning, remainingTime, dispatch]);

  // Handle page unload/close to ensure timer state is preserved
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (isRunning) {
        // Timer state will be persisted by Redux Persist automatically
        // No additional action needed
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [isRunning]);
};