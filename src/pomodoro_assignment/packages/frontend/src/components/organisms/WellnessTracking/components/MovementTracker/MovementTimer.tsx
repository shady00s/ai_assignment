import React, { useState, useEffect, useCallback } from 'react';
import styled, { keyframes } from 'styled-components';

interface MovementTimerProps {
  isActive: boolean;
  onStart: () => void;
  onEnd: (duration: number) => void;
  disabled?: boolean;
  compact?: boolean;
}

const pulseAnimation = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.05);
    opacity: 0.8;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

const TimerContainer = styled.div<{ compact?: boolean }>`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.md};
  padding: ${({ theme, compact }) => compact ? theme.spacing.sm : theme.spacing.md};
  background: ${({ theme }) => theme.colors.neutral[50]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border: 2px solid ${({ theme, isActive }) => isActive ? theme.colors.sunriseOrange : theme.colors.neutral[200]};
  transition: all 0.3s ease;
`;

const TimerDisplay = styled.div<{ compact?: boolean; isActive?: boolean }>`
  font-size: ${({ compact }) => compact ? '24px' : '32px'};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme, isActive }) => isActive ? theme.colors.sunriseOrange : theme.colors.neutral[700]};
  font-variant-numeric: tabular-nums;
  min-width: ${({ compact }) => compact ? '80px' : '120px'};
  text-align: center;
  animation: ${({ isActive }) => isActive ? `${pulseAnimation} 2s ease-in-out infinite` : 'none'};
`;

const TimerLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  text-align: center;
`;

const ControlButton = styled.button<{ variant?: 'start' | 'stop' | 'pause' }>`
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.lg};
  border: none;
  border-radius: ${({ theme }) => theme.borderRadius.full};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  min-width: 100px;
  justify-content: center;

  ${({ variant = 'start', theme }) => {
    switch (variant) {
      case 'start':
        return `
          background-color: ${theme.colors.success};
          color: white;
          &:hover {
            background-color: ${theme.colors.success}dd;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          }
        `;
      case 'stop':
        return `
          background-color: ${theme.colors.error};
          color: white;
          &:hover {
            background-color: ${theme.colors.error}dd;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          }
        `;
      case 'pause':
        return `
          background-color: ${theme.colors.warning};
          color: white;
          &:hover {
            background-color: ${theme.colors.warning}dd;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          }
        `;
    }
  }}

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none !important;
  }
`;

const ProgressBar = styled.div<{ compact?: boolean }>`
  width: ${({ compact }) => compact ? '100%' : '200px'};
  height: 4px;
  background-color: ${({ theme }) => theme.colors.neutral[200]};
  border-radius: 2px;
  overflow: hidden;
  margin-top: ${({ theme }) => theme.spacing.sm};
`;

const ProgressFill = styled.div<{ progress: number }>`
  height: 100%;
  background: linear-gradient(90deg, ${({ theme }) => theme.colors.sunriseOrange}, ${({ theme }) => theme.colors.warning});
  border-radius: 2px;
  width: ${({ progress }) => progress}%;
  transition: width 1s linear;
`;

const StatusIndicator = styled.div<{ isActive?: boolean }>`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme, isActive }) => isActive ? theme.colors.sunriseOrange : theme.colors.neutral[500]};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const StatusDot = styled.div<{ isActive?: boolean }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${({ theme, isActive }) => isActive ? theme.colors.success : theme.colors.neutral[300]};
  animation: ${({ isActive }) => isActive ? `${pulseAnimation} 1.5s ease-in-out infinite` : 'none'};
`;

export const MovementTimer: React.FC<MovementTimerProps> = ({
  isActive,
  onStart,
  onEnd,
  disabled = false,
  compact = false,
}) => {
  const [seconds, setSeconds] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);

  // Format time display
  const formatTime = (totalSeconds: number): string => {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  // Handle timer start
  const handleStart = useCallback(() => {
    if (disabled) return;

    const now = Date.now();
    setStartTime(now);
    setSeconds(0);
    onStart();
  }, [disabled, onStart]);

  // Handle timer stop
  const handleStop = useCallback(() => {
    if (!startTime) return;

    const duration = Math.round((Date.now() - startTime) / 1000);
    setSeconds(0);
    setStartTime(null);
    onEnd(duration);
  }, [startTime, onEnd]);

  // Update timer
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isActive && startTime) {
      interval = setInterval(() => {
        setSeconds(Math.round((Date.now() - startTime) / 1000));
      }, 100);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isActive, startTime]);

  // Calculate progress for visual indicator (based on a 5-minute target)
  const targetSeconds = 5 * 60; // 5 minutes
  const progress = Math.min(100, (seconds / targetSeconds) * 100);

  return (
    <TimerContainer compact={compact} isActive={isActive}>
      <TimerDisplay compact={compact} isActive={isActive}>
        {formatTime(seconds)}
      </TimerDisplay>

      <TimerLabel>
        {isActive ? 'Movement in progress' : 'Start a movement break'}
      </TimerLabel>

      {isActive && (
        <ProgressBar compact={compact}>
          <ProgressFill progress={progress} />
        </ProgressBar>
      )}

      <StatusIndicator isActive={isActive}>
        <StatusDot isActive={isActive} />
        {isActive ? 'Active' : 'Ready'}
      </StatusIndicator>

      <ControlButton
        variant={isActive ? 'stop' : 'start'}
        onClick={isActive ? handleStop : handleStart}
        disabled={disabled}
        aria-label={isActive ? 'Stop movement timer' : 'Start movement timer'}
      >
        {isActive ? (
          <>
            ⏹️ End Break
          </>
        ) : (
          <>
            ▶️ Start Break
          </>
        )}
      </ControlButton>
    </TimerContainer>
  );
};