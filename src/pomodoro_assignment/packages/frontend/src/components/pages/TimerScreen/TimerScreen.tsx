import React, { useEffect, useRef, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { CircularTimer } from '../../organisms/CircularTimer';
import {
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
  startTimer,
  pauseTimer,
  skipSession,
  setSessionType,
  decrementTime,
} from '../../../store';

interface TimerScreenProps {
  className?: string;
}

export const TimerScreen: React.FC<TimerScreenProps> = ({ className }) => {
  const dispatch = useDispatch();
  const timerState = useSelector(selectTimerState);
  const isRunning = useSelector(selectIsRunning);
  const isPaused = useSelector(selectIsPaused);
  const remainingTime = useSelector(selectRemainingTime);
  const totalTime = useSelector(selectTotalTime);
  const sessionType = useSelector(selectSessionType);
  const sessionsCompleted = useSelector(selectSessionsCompleted);
  const workDuration = useSelector(selectWorkDuration);
  const shortBreakDuration = useSelector(selectShortBreakDuration);
  const longBreakDuration = useSelector(selectLongBreakDuration);

  // Refs for timer management
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Calculate progress for circular timer - memoized for performance
  const progress = useMemo(() => {
    return totalTime > 0 ? (totalTime - remainingTime) / totalTime : 0;
  }, [totalTime, remainingTime]);

  // Timer effect - decrement time every second when running
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
    }

    // Auto-complete when timer reaches zero
    if (isRunning && remainingTime === 0) {
      handleComplete();
    }

    // Cleanup function
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isRunning, remainingTime, dispatch]);

  // Handle timer control actions with proper error handling
  const handleStart = () => {
    dispatch(startTimer());
  };

  const handlePause = () => {
    dispatch(pauseTimer());
  };

  const handleComplete = () => {
    // Dispatch skip action for now (we'll implement complete later)
    dispatch(skipSession());
  };

  const handleSkip = () => {
    dispatch(skipSession());
  };

  const handleSessionTypeChange = (type: 'POMODORO' | 'SHORT_BREAK' | 'LONG_BREAK') => {
    if (!isRunning) {
      dispatch(setSessionType(type));
    }
  };

  // Session type configuration
  const sessionTypes = useMemo(() => [
    {
      type: 'POMODORO' as const,
      label: 'Pomodoro',
      icon: '🍅',
      color: '#7FA870',
    },
    {
      type: 'SHORT_BREAK' as const,
      label: 'Short Break',
      icon: '☕',
      color: '#F4A261',
    },
    {
      type: 'LONG_BREAK' as const,
      label: 'Long Break',
      icon: '🌿',
      color: '#E9C46A',
    },
  ], []);

  // Format duration for display
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    return `${mins}min`;
  };

  return (
    <div className={className}>
      {/* Session Type Selector - Enhanced with accessibility */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '12px',
          marginBottom: '40px',
          flexWrap: 'wrap',
        }}
      >
        {sessionTypes.map(({ type, label, icon, color }) => (
          <button
            key={type}
            onClick={() => handleSessionTypeChange(type)}
            disabled={isRunning}
            style={{
              padding: '12px 20px',
              background: sessionType === type ? color : 'transparent',
              color: sessionType === type ? 'white' : '#8B7D7B',
              border: `2px solid ${color}`,
              borderRadius: '12px',
              fontSize: '14px',
              fontWeight: '600',
              cursor: isRunning ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease',
              opacity: isRunning ? 0.6 : 1,
            }}
          >
            <span style={{ marginRight: '6px' }}>{icon}</span> {label}
          </button>
        ))}
      </div>

      {/* Main Timer Display */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          marginBottom: '40px',
        }}
      >
        <div
          style={{
            padding: '40px',
            background: 'linear-gradient(135deg, #F8F9FA 0%, #E8D8C8 100%)',
            boxShadow: '0 12px 40px rgba(0, 0, 0, 0.08)',
            border: '1px solid rgba(127, 168, 112, 0.1)',
            borderRadius: '32px',
            backdropFilter: 'blur(10px)',
          }}
        >
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '32px',
          }}>
            {/* Enhanced Circular Timer */}
            <div style={{
              position: 'relative',
              filter: 'drop-shadow(0 8px 24px rgba(127, 168, 112, 0.15))',
            }}>
              <CircularTimer
                size={320}
                strokeWidth={14}
                showControls={false}
                progress={progress}
                remainingTime={remainingTime}
                sessionType={sessionType}
              />
            </div>

            {/* Session Counter with enhanced styling */}
            <div style={{
              textAlign: 'center',
              background: 'linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%)',
              padding: '16px 32px',
              borderRadius: '20px',
              color: 'white',
              boxShadow: '0 4px 16px rgba(127, 168, 112, 0.25)',
            }}>
              <div style={{
                fontSize: '13px',
                opacity: 0.9,
                marginBottom: '4px',
                fontFamily: 'Inter, sans-serif',
                fontWeight: '500',
              }}>
                Sessions Completed Today
              </div>
              <div
                style={{
                  fontSize: '28px',
                  fontWeight: 'bold',
                  fontFamily: 'Lora, serif',
                }}
              >
                {sessionsCompleted}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Timer Controls with better accessibility */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '16px',
          marginBottom: '48px',
          flexWrap: 'wrap',
        }}
      >
        {!isRunning ? (
          <button
            onClick={handleStart}
            style={{
              padding: '18px 36px',
              fontSize: '18px',
              minWidth: '150px',
              height: '60px',
              background: 'linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '16px',
              boxShadow: '0 6px 20px rgba(127, 168, 112, 0.3)',
              cursor: 'pointer',
              fontWeight: '600',
              transition: 'all 0.2s ease',
            }}
          >
            <span style={{ marginRight: '8px' }}>▶️</span> {isPaused ? 'Resume' : 'Start'}
          </button>
        ) : (
          <button
            onClick={handlePause}
            style={{
              padding: '18px 36px',
              fontSize: '18px',
              minWidth: '150px',
              height: '60px',
              background: 'linear-gradient(135deg, #F4A261 0%, #F5B789 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '16px',
              boxShadow: '0 6px 20px rgba(244, 162, 97, 0.3)',
              cursor: 'pointer',
              fontWeight: '600',
              transition: 'all 0.2s ease',
            }}
          >
            <span style={{ marginRight: '8px' }}>⏸️</span> Pause
          </button>
        )}

        <button
          onClick={handleSkip}
          disabled={!isRunning && !isPaused}
          style={{
            padding: '18px 36px',
            fontSize: '18px',
            minWidth: '150px',
            height: '60px',
            backgroundColor: 'transparent',
            color: '#8B7D7B',
            border: '2px solid #D4C4B0',
            borderRadius: '16px',
            cursor: (!isRunning && !isPaused) ? 'not-allowed' : 'pointer',
            fontWeight: '600',
            transition: 'all 0.2s ease',
            opacity: (!isRunning && !isPaused) ? 0.6 : 1,
          }}
        >
          <span style={{ marginRight: '8px' }}>⏭️</span> Skip
        </button>

        <button
          onClick={handleComplete}
          disabled={!isRunning && !isPaused}
          style={{
            padding: '18px 36px',
            fontSize: '18px',
            minWidth: '150px',
            height: '60px',
            background: 'linear-gradient(135deg, #C85A5A 0%, #D57A7A 100%)',
            color: 'white',
            border: 'none',
            borderRadius: '16px',
            cursor: (!isRunning && !isPaused) ? 'not-allowed' : 'pointer',
            fontWeight: '600',
            transition: 'all 0.2s ease',
            opacity: (!isRunning && !isPaused) ? 0.6 : 1,
          }}
        >
          <span style={{ marginRight: '8px' }}>✅</span> Complete
        </button>
      </div>

      {/* Enhanced Quick Stats */}
      <section
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '20px',
          maxWidth: '680px',
          margin: '0 auto',
        }}
      >
        <div
          style={{
            textAlign: 'center',
            padding: '24px',
            background: 'linear-gradient(135deg, #F8F9FA 0%, #F0E6DC 100%)',
            border: '1px solid rgba(127, 168, 112, 0.1)',
            borderRadius: '20px',
            transition: 'transform 0.2s ease, box-shadow 0.2s ease',
          }}
        >
          <div style={{
            fontSize: '24px',
            marginBottom: '8px',
          }}>
            🍅
          </div>
          <div style={{
            fontSize: '13px',
            color: '#8B7D7B',
            marginBottom: '6px',
            fontFamily: 'Inter, sans-serif',
            fontWeight: '500',
          }}>
            Focus Duration
          </div>
          <div style={{
            fontSize: '18px',
            fontWeight: 'bold',
            color: '#2C3E50',
            fontFamily: 'Lora, serif',
          }}>
            {formatDuration(Number(workDuration))}
          </div>
        </div>

        <div
          style={{
            textAlign: 'center',
            padding: '24px',
            background: 'linear-gradient(135deg, #F8F9FA 0%, #F0E6DC 100%)',
            border: '1px solid rgba(244, 162, 97, 0.1)',
            borderRadius: '20px',
            transition: 'transform 0.2s ease, box-shadow 0.2s ease',
          }}
        >
          <div style={{
            fontSize: '24px',
            marginBottom: '8px',
          }}>
            ☕
          </div>
          <div style={{
            fontSize: '13px',
            color: '#8B7D7B',
            marginBottom: '6px',
            fontFamily: 'Inter, sans-serif',
            fontWeight: '500',
          }}>
            Short Break
          </div>
          <div style={{
            fontSize: '18px',
            fontWeight: 'bold',
            color: '#2C3E50',
            fontFamily: 'Lora, serif',
          }}>
            {formatDuration(Number(shortBreakDuration))}
          </div>
        </div>

        <div
          style={{
            textAlign: 'center',
            padding: '24px',
            background: 'linear-gradient(135deg, #F8F9FA 0%, #F0E6DC 100%)',
            border: '1px solid rgba(233, 196, 106, 0.1)',
            borderRadius: '20px',
            transition: 'transform 0.2s ease, box-shadow 0.2s ease',
          }}
        >
          <div style={{
            fontSize: '24px',
            marginBottom: '8px',
          }}>
            🌿
          </div>
          <div style={{
            fontSize: '13px',
            color: '#8B7D7B',
            marginBottom: '6px',
            fontFamily: 'Inter, sans-serif',
            fontWeight: '500',
          }}>
            Long Break
          </div>
          <div style={{
            fontSize: '18px',
            fontWeight: 'bold',
            color: '#2C3E50',
            fontFamily: 'Lora, serif',
          }}>
            {formatDuration(Number(longBreakDuration))}
          </div>
        </div>
      </section>
    </div>
  );
};

export type { TimerScreenProps };