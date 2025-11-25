import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

interface MovementTrackerProps {
  movementBreaks: number;
  movementMinutes: number;
  stepsCount?: number;
  dailyGoal: number;
  onStartBreak?: () => void;
  onEndBreak?: (duration: number) => void;
  onLogActivity?: (minutes: number, type: string) => void;
  isLoading?: boolean;
  compact?: boolean;
  className?: string;
}

const MovementContainer = styled.div<{ $compact: boolean }>`
  background: rgba(255, 255, 255, 0.8);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ $compact, theme }) => $compact ? theme.spacing.mobile.sm : theme.spacing.mobile.md};
  border: 1px solid rgba(230, 126, 34, 0.2);
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ $compact, theme }) => $compact ? theme.spacing.tablet.sm : theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ $compact }) => $compact ? '12px' : '20px'};
  }
`;

const MovementTitle = styled.h4`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #E67E22;
  margin: 0 0 ${({ theme }) => theme.spacing.mobile.sm} 0;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
`;

const MovementStats = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
`;

const MovementStat = styled.div`
  text-align: center;
`;

const StatValue = styled.div<{ $color: string }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ $color }) => $color};
  margin-bottom: 2px;
`;

const StatLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: #8B7D7B;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const ProgressBar = styled.div<{ $color: string }>`
  width: 100%;
  height: 6px;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
`;

const ProgressFill = styled.div<{ $progress: number; $color: string }>`
  height: 100%;
  background: ${({ $color }) => $color};
  border-radius: inherit;
  width: ${({ $progress }) => $progress}%;
  transition: width 0.5s ease-in-out;
`;

const MovementControls = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  justify-content: center;
`;

const MovementButton = styled.button<{ $variant: 'primary' | 'secondary' }>`
  background: ${({ $variant }) => $variant === 'primary' ? '#E67E22' : 'transparent'};
  color: ${({ $variant }) => $variant === 'primary' ? 'white' : '#E67E22'};
  border: 2px solid #E67E22;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.mobile.xs} ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: ${({ $variant }) => $variant === 'primary' ? '0 4px 12px rgba(230, 126, 34, 0.3)' : 'none'};
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
`;

export const MovementTracker: React.FC<MovementTrackerProps> = ({
  movementBreaks,
  movementMinutes,
  stepsCount = 0,
  dailyGoal,
  onStartBreak,
  onEndBreak,
  onLogActivity,
  isLoading = false,
  compact = false,
  className,
}) => {
  const [isOnBreak, setIsOnBreak] = useState(false);
  const [breakStartTime, setBreakStartTime] = useState<number | null>(null);
  const [currentBreakDuration, setCurrentBreakDuration] = useState(0);

  const progress = Math.min((movementBreaks / dailyGoal) * 100, 100);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isOnBreak && breakStartTime) {
      interval = setInterval(() => {
        setCurrentBreakDuration(Math.floor((Date.now() - breakStartTime) / 1000 / 60));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isOnBreak, breakStartTime]);

  const handleStartBreak = () => {
    setIsOnBreak(true);
    setBreakStartTime(Date.now());
    setCurrentBreakDuration(0);
    onStartBreak?.();
  };

  const handleEndBreak = () => {
    if (breakStartTime) {
      const duration = Math.floor((Date.now() - breakStartTime) / 1000 / 60);
      setIsOnBreak(false);
      setBreakStartTime(null);
      setCurrentBreakDuration(0);
      onEndBreak?.(duration);
    }
  };

  const quickActivities = [
    { minutes: 2, label: '2 min stretch', type: 'stretch' },
    { minutes: 5, label: '5 min walk', type: 'walk' },
    { minutes: 10, label: '10 min exercise', type: 'exercise' },
  ];

  return (
    <MovementContainer $compact={compact} className={className}>
      <MovementTitle>🚶 Movement</MovementTitle>

      <MovementStats>
        <MovementStat>
          <StatValue $color="#E67E22">{movementBreaks}</StatValue>
          <StatLabel>Breaks Today</StatLabel>
        </MovementStat>
        <MovementStat>
          <StatValue $color="#27AE60">{movementMinutes}m</StatValue>
          <StatLabel>Total Minutes</StatLabel>
        </MovementStat>
        {stepsCount > 0 && (
          <MovementStat>
            <StatValue $color="#3498DB">{stepsCount.toLocaleString()}</StatValue>
            <StatLabel>Steps</StatLabel>
          </MovementStat>
        )}
        <MovementStat>
          <StatValue $color="#9B59B6">{dailyGoal}</StatValue>
          <StatLabel>Daily Goal</StatLabel>
        </MovementStat>
      </MovementStats>

      <ProgressBar $color="#E67E22">
        <ProgressFill $progress={progress} $color="#E67E22" />
      </ProgressBar>

      {isOnBreak && (
        <div style={{
          textAlign: 'center',
          fontSize: '12px',
          color: '#E67E22',
          fontWeight: 'bold',
          marginBottom: '8px',
        }}>
          🏃 Current Break: {currentBreakDuration}m
        </div>
      )}

      <MovementControls>
        {!isOnBreak ? (
          <MovementButton
            $variant="primary"
            onClick={handleStartBreak}
            disabled={isLoading}
          >
            🏃 Start Break
          </MovementButton>
        ) : (
          <MovementButton
            $variant="secondary"
            onClick={handleEndBreak}
            disabled={isLoading}
          >
            ✅ End Break ({currentBreakDuration}m)
          </MovementButton>
        )}
      </MovementControls>

      {!compact && (
        <div style={{
          marginTop: '8px',
          display: 'flex',
          gap: '4px',
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}>
          {quickActivities.map((activity) => (
            <MovementButton
              key={activity.type}
              $variant="secondary"
              onClick={() => onLogActivity?.(activity.minutes, activity.type)}
              disabled={isLoading}
              style={{
                fontSize: '10px',
                padding: '2px 6px',
              }}
            >
              {activity.label}
            </MovementButton>
          ))}
        </div>
      )}
    </MovementContainer>
  );
};

export type { MovementTrackerProps };