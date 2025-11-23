import React, { useEffect, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import styled from 'styled-components';
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
} from '../../../store';

const TimerContainer = styled.div`
  padding: ${({ theme }) => theme.spacing.mobile.md};
  max-width: 100%;
  margin: 0 auto;

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme }) => theme.spacing.xl};
    max-width: 1200px;
  }
`;

const SessionTypeSelector = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xl};
  flex-wrap: wrap;

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.md};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: ${({ theme }) => theme.spacing.lg};
    margin-bottom: ${({ theme }) => theme.spacing['2xl']};
  }
`;

const SessionButton = styled.button<{ $active: boolean; $color: string; $disabled: boolean }>`
  padding: ${({ theme }) => theme.spacing.mobile.sm} ${({ theme }) => theme.spacing.mobile.md};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  background: ${({ $active, $color }) => $active ? $color : 'transparent'};
  color: ${({ $active, $color }) => $active ? 'white' : '#8B7D7B'};
  border: 2px solid ${({ $color }) => $color};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  cursor: ${({ $disabled }) => $disabled ? 'not-allowed' : 'pointer'};
  opacity: ${({ $disabled }) => $disabled ? 0.6 : 1};
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.sm} ${({ theme }) => theme.spacing.tablet.md};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.lg};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    gap: ${({ theme }) => theme.spacing.sm};
  }

  &:hover:not(:disabled) {
    background: ${({ $active, $color }) => $active ? $color : `${$color}20`};
  }
`;

const TimerDisplay = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xl};

  ${({ theme }) => theme.mediaQueries.tablet} {
    margin-bottom: ${({ theme }) => theme.spacing.tablet.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    margin-bottom: ${({ theme }) => theme.spacing['2xl']};
  }
`;

const TimerCard = styled.div`
  padding: ${({ theme }) => theme.spacing.mobile.lg};
  background: linear-gradient(135deg, #F8F9FA 0%, #E8D8C8 100%);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(127, 168, 112, 0.1);
  border-radius: ${({ theme }) => theme.spacing.mobile['3xl']};
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.xl};
    border-radius: ${({ theme }) => theme.spacing.tablet['3xl']};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 40px;
    border-radius: 32px;
  }
`;

const TimerContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xl};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 32px;
  }
`;

const CircularTimerWrapper = styled.div`
  position: relative;
  filter: drop-shadow(0 8px 24px rgba(127, 168, 112, 0.15));
`;

const SessionCounter = styled.div`
  text-align: center;
  background: linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%);
  padding: ${({ theme }) => theme.spacing.mobile.md} ${({ theme }) => theme.spacing.mobile.xl};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  color: white;
  box-shadow: 0 4px 16px rgba(127, 168, 112, 0.25);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md} ${({ theme }) => theme.spacing.tablet.xl};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 16px 32px;
    border-radius: 20px;
  }
`;

const CounterLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  opacity: 0.9;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xs};
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 13px;
  }
`;

const CounterValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet['3xl']};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 28px;
  }
`;

const TimerControls = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.md};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xl};
  flex-wrap: wrap;

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.lg};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: ${({ theme }) => theme.spacing.lg};
    margin-bottom: 48px;
  }
`;

const ControlButton = styled.button<{ $variant: 'primary' | 'secondary' | 'danger' | 'warning'; $disabled?: boolean }>`
  padding: ${({ theme }) => theme.spacing.mobile.lg} ${({ theme }) => theme.spacing.mobile.xl};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  min-width: ${({ theme }) => theme.spacing.mobile['2xl']};
  height: 60px;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  cursor: ${({ $disabled }) => $disabled ? 'not-allowed' : 'pointer'};
  opacity: ${({ $disabled }) => $disabled ? 0.6 : 1};
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.lg} ${({ theme }) => theme.spacing.tablet.xl};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.lg};
    min-width: ${({ theme }) => theme.spacing.tablet['2xl']};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 18px 36px;
    font-size: 18px;
    min-width: 150px;
    gap: 8px;
  }

  ${({ $variant }) => {
    switch ($variant) {
      case 'primary':
        return `
          background: linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%);
          color: white;
          border: none;
          box-shadow: 0 6px 20px rgba(127, 168, 112, 0.3);
        `;
      case 'warning':
        return `
          background: linear-gradient(135deg, #F4A261 0%, #F5B789 100%);
          color: white;
          border: none;
          box-shadow: 0 6px 20px rgba(244, 162, 97, 0.3);
        `;
      case 'danger':
        return `
          background: linear-gradient(135deg, #C85A5A 0%, #D57A7A 100%);
          color: white;
          border: none;
        `;
      case 'secondary':
        return `
          background: transparent;
          color: #8B7D7B;
          border: 2px solid #D4C4B0;
        `;
      default:
        return '';
    }
  }}

  &:hover:not(:disabled) {
    transform: translateY(-1px);
  }
`;

const StatsGrid = styled.section`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: ${({ theme }) => theme.spacing.mobile.lg};
  max-width: 680px;
  margin: 0 auto;

  ${({ theme }) => theme.mediaQueries.tablet} {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
  }
`;

const StatCard = styled.div<{ $borderColor: string }>`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.mobile.lg};
  background: linear-gradient(135deg, #F8F9FA 0%, #F0E6DC 100%);
  border: 1px solid ${({ $borderColor }) => $borderColor};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 24px;
  }

  &:hover {
    transform: translateY(-2px);
  }
`;

const StatIcon = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile['2xl']};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 24px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 24px;
  }
`;

const StatLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #8B7D7B;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 13px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 13px;
  }
`;

const StatValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: #2C3E50;
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 18px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 18px;
  }
`;

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

  // Calculate progress for circular timer - memoized for performance
  const progress = useMemo(() => {
    return totalTime > 0 ? (totalTime - remainingTime) / totalTime : 0;
  }, [totalTime, remainingTime]);

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

  // Auto-complete when timer reaches zero
  useEffect(() => {
    if (isRunning && remainingTime === 0) {
      handleComplete();
    }
  }, [isRunning, remainingTime, dispatch]);

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
    <TimerContainer className={className}>
      {/* Session Type Selector */}
      <SessionTypeSelector>
        {sessionTypes.map(({ type, label, icon, color }) => (
          <SessionButton
            key={type}
            onClick={() => handleSessionTypeChange(type)}
            $active={sessionType === type}
            $color={color}
            $disabled={isRunning}
          >
            <span>{icon}</span> {label}
          </SessionButton>
        ))}
      </SessionTypeSelector>

      {/* Main Timer Display */}
      <TimerDisplay>
        <TimerCard>
          <TimerContent>
            {/* Circular Timer */}
            <CircularTimerWrapper>
              <CircularTimer
                size={280}
                strokeWidth={12}
                showControls={false}
                progress={progress}
                remainingTime={remainingTime}
                sessionType={sessionType}
              />
            </CircularTimerWrapper>

            {/* Session Counter */}
            <SessionCounter>
              <CounterLabel>Sessions Completed Today</CounterLabel>
              <CounterValue>{sessionsCompleted}</CounterValue>
            </SessionCounter>
          </TimerContent>
        </TimerCard>
      </TimerDisplay>

      {/* Timer Controls */}
      <TimerControls>
        {!isRunning ? (
          <ControlButton
            onClick={handleStart}
            $variant="primary"
          >
            <span>▶️</span> {isPaused ? 'Resume' : 'Start'}
          </ControlButton>
        ) : (
          <ControlButton
            onClick={handlePause}
            $variant="warning"
          >
            <span>⏸️</span> Pause
          </ControlButton>
        )}

        <ControlButton
          onClick={handleSkip}
          $variant="secondary"
          $disabled={!isRunning && !isPaused}
        >
          <span>⏭️</span> Skip
        </ControlButton>

        <ControlButton
          onClick={handleComplete}
          $variant="danger"
          $disabled={!isRunning && !isPaused}
        >
          <span>✅</span> Complete
        </ControlButton>
      </TimerControls>

      {/* Quick Stats */}
      <StatsGrid>
        <StatCard $borderColor="rgba(127, 168, 112, 0.1)">
          <StatIcon>🍅</StatIcon>
          <StatLabel>Focus Duration</StatLabel>
          <StatValue>{formatDuration(Number(workDuration))}</StatValue>
        </StatCard>

        <StatCard $borderColor="rgba(244, 162, 97, 0.1)">
          <StatIcon>☕</StatIcon>
          <StatLabel>Short Break</StatLabel>
          <StatValue>{formatDuration(Number(shortBreakDuration))}</StatValue>
        </StatCard>

        <StatCard $borderColor="rgba(233, 196, 106, 0.1)">
          <StatIcon>🌿</StatIcon>
          <StatLabel>Long Break</StatLabel>
          <StatValue>{formatDuration(Number(longBreakDuration))}</StatValue>
        </StatCard>
      </StatsGrid>
    </TimerContainer>
  );
};

export type { TimerScreenProps };