import React from 'react';
import styled from 'styled-components';
import { Task } from '../../../../../types';

interface SessionType {
  POMODORO: 'POMODORO';
  SHORT_BREAK: 'SHORT_BREAK';
  LONG_BREAK: 'LONG_BREAK';
  CUSTOM: 'CUSTOM';
}

interface TaskProgressProps {
  task?: Task;
  sessionCount?: number;
  totalSessions?: number;
  showTimeEstimate?: boolean;
  className?: string;
  // Real-time progress props
  isTimerRunning?: boolean;
  sessionType?: string;
  sessionProgress?: number; // 0-1 progress of current session
}

const ProgressContainer = styled.div`
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
  backdrop-filter: blur(10px);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.mobile.md};
  border: 1px solid rgba(127, 168, 112, 0.15);
  box-shadow: 0 4px 20px rgba(127, 168, 112, 0.08);
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg,
      #8B5CF6 0%,
      #3B82F6 25%,
      #F59E0B 75%,
      #10B981 100%);
    opacity: 0.8;
  }

  /* Dark mode styles */
  .dark-mode & {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(51, 65, 85, 0.9) 100%) !important;
    border-color: rgba(127, 168, 112, 0.25) !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
    box-shadow: 0 6px 30px rgba(127, 168, 112, 0.12);
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 8px 40px rgba(127, 168, 112, 0.15);
  }
`;

const ProgressHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.md};
`;

const ProgressLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #4B5563;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  display: flex;
  align-items: center;
  gap: 8px;

  /* Dark mode styles */
  .dark-mode & {
    color: #9CA3AF !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.base};
    gap: 10px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 15px;
    gap: 12px;
  }
`;

const ProgressValue = styled.div<{ $color?: string }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ $color }) => $color || '#2C3E50'};
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  background: ${({ $color }) => $color ? `${$color}15` : 'transparent'};
  padding: 4px 8px;
  border-radius: 6px;
  min-width: 50px;
  text-align: center;
  transition: all 0.3s ease;

  /* Dark mode styles */
  .dark-mode & {
    color: ${({ $color }) => $color || '#F1F5F9'} !important;
    background: ${({ $color }) => $color ? `${$color}25` : 'transparent'} !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.base};
    padding: 6px 12px;
    min-width: 60px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 16px;
    padding: 8px 16px;
    min-width: 70px;
  }
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: rgba(229, 231, 235, 0.8);
  border-radius: 6px;
  overflow: hidden;
  position: relative;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);

  /* Dark mode styles */
  .dark-mode & {
    background: rgba(75, 85, 99, 0.5) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    height: 10px;
    border-radius: 8px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    height: 12px;
    border-radius: 10px;
  }
`;

const ProgressFill = styled.div<{ $progress: number; $color?: string; $isActive?: boolean }>`
  height: 100%;
  background: ${({ $color }) => $color ?
    `linear-gradient(90deg, ${$color} 0%, ${$color}DD 50%, ${$color} 100%)` :
    'linear-gradient(90deg, #7FA870 0%, #8FBC8F 50%, #7FA870 100%)'};
  border-radius: inherit;
  width: ${({ $progress }) => $progress}%;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  box-shadow: ${({ $isActive }) => $isActive ?
    '0 0 20px rgba(127, 168, 112, 0.4), inset 0 1px 2px rgba(255,255,255,0.3)' :
    'inset 0 1px 2px rgba(0,0,0,0.1)'};

  /* Shimmer effect for active sessions */
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg,
      transparent,
      ${({ $color, $isActive }) => $isActive && $color ? `${$color}40` : 'rgba(255,255,255,0.4)'},
      transparent
    );
    animation: ${({ $isActive }) => $isActive ? 'shimmer 2s infinite' : 'none'};
  }

  /* Pulse effect for active sessions */
  ${({ $isActive }) => $isActive && `
    animation: pulse 2s ease-in-out infinite;
  `}

  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
  }
`;

const CurrentSessionOverlay = styled.div<{ $progress: number; $isVisible: boolean }>`
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  width: ${({ $progress }) => $progress}%;
  background: linear-gradient(90deg,
    transparent 0%,
    rgba(59, 130, 246, 0.3) 30%,
    rgba(59, 130, 246, 0.5) 70%,
    transparent 100%
  );
  border-radius: inherit;
  opacity: ${({ $isVisible }) => $isVisible ? 1 : 0};
  transition: opacity 0.3s ease, width 0.2s ease;
  pointer-events: none;
  animation: currentSessionPulse 3s ease-in-out infinite;

  @keyframes currentSessionPulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
  }
`;

const ProgressDetails = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: ${({ theme }) => theme.spacing.mobile.xs};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: #A8968E;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    margin-top: ${({ theme }) => theme.spacing.tablet.xs};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.xs};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    margin-top: 6px;
    font-size: 11px;
  }
`;

const TimeEstimate = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  color: #7FA870;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.xs};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 4px;
  }
`;

const CompletionIndicator = styled.div<{ $status: 'ahead' | 'on-track' | 'behind' }>`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  color: ${({ $status }) => {
    switch ($status) {
      case 'ahead': return '#7FA870';
      case 'on-track': return '#F4A261';
      case 'behind': return '#C85A5A';
      default: return '#F4A261';
    }
  }};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.xs};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 4px;
  }
`;

// Utility functions for enhanced progress display
const getProgressColor = (progress: number): string => {
  if (progress === 100) return '#10B981'; // Green - complete
  if (progress >= 75) return '#F59E0B'; // Orange - almost done
  if (progress >= 50) return '#3B82F6'; // Blue - good progress
  if (progress >= 25) return '#8B5CF6'; // Purple - making progress
  return '#8B5CF6'; // Purple - just starting
};

const getStatusColor = (progress: number): string => {
  if (progress === 100) return '#10B981'; // ahead
  if (progress >= 75) return '#F59E0B'; // on-track
  if (progress >= 50) return '#3B82F6'; // on-track
  return '#C85A5A'; // behind
};

const formatTimeRemaining = (minutes: number): string => {
  if (minutes < 60) return `${minutes}min`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return mins > 0 ? `${hours}h ${mins}min` : `${hours}h`;
};

const getSessionTimeRemaining = (sessionProgress: number): string => {
  const totalSeconds = 25 * 60; // 25 minutes in seconds
  const remainingSeconds = totalSeconds * (1 - sessionProgress);
  const remainingMinutes = Math.ceil(remainingSeconds / 60);
  return `${Math.floor(remainingSeconds / 60)}:${(remainingSeconds % 60).toString().padStart(2, '0')}`;
};

export const TaskProgress: React.FC<TaskProgressProps> = ({
  task,
  sessionCount = 0,
  totalSessions = 4,
  showTimeEstimate = true,
  className,
  isTimerRunning = false,
  sessionType = 'POMODORO',
  sessionProgress = 0,
}) => {
  if (!task) {
    return null;
  }

  // Calculate real-time progress including current session
  const currentSessionContribution = (isTimerRunning && sessionType === 'POMODORO') ? sessionProgress : 0;
  const effectiveCompletedPomodoros = task.completedPomodoros + currentSessionContribution;
  const taskProgress = Math.min((effectiveCompletedPomodoros / task.estimatedPomodoros) * 100, 100);

  const remainingPomodoros = Math.max(task.estimatedPomodoros - effectiveCompletedPomodoros, 0);
  const estimatedMinutes = Math.ceil(remainingPomodoros * 25);

  // Get dynamic colors based on progress
  const progressColor = getProgressColor(taskProgress);
  const statusColor = getStatusColor(taskProgress);

  // Enhanced completion status with motivational messages
  const getCompletionStatus = () => {
    if (taskProgress === 100) return { status: 'ahead' as const, text: 'Task Complete! 🎉', icon: '✅', message: 'Great job!' };
    if (taskProgress >= 90) return { status: 'ahead' as const, text: 'Final stretch!', icon: '🏁', message: 'Almost there!' };
    if (taskProgress >= 75) return { status: 'ahead' as const, text: 'Almost done', icon: '🎯', message: 'Keep going!' };
    if (taskProgress >= 60) return { status: 'on-track' as const, text: 'Good progress', icon: '📈', message: 'You\'re doing great!' };
    if (taskProgress >= 40) return { status: 'on-track' as const, text: 'On track', icon: '📊', message: 'Steady progress!' };
    if (taskProgress >= 20) return { status: 'on-track' as const, text: 'Building momentum', icon: '🚀', message: 'Keep it up!' };
    return { status: 'behind' as const, text: 'Just starting', icon: '🌱', message: 'You got this!' };
  };

  const completionStatus = getCompletionStatus();

  // Calculate completion time estimate
  const getCompletionETA = (): string => {
    if (remainingPomodoros === 0) return 'Complete!';
    const now = new Date();
    const totalMinutes = remainingPomodoros * 25; // 25 min per pomodoro
    const eta = new Date(now.getTime() + totalMinutes * 60 * 1000);
    return `Done by ${eta.getHours()}:${(eta.getMinutes()).toString().padStart(2, '0')}`;
  };

  return (
    <ProgressContainer className={className}>
      <ProgressHeader>
        <ProgressLabel>
          📊 Task Progress
          {isTimerRunning && sessionType === 'POMODORO' && (
            <span style={{
              fontSize: '12px',
              color: '#3B82F6',
              marginLeft: '4px',
              fontWeight: 'normal'
            }}>
              • Active
            </span>
          )}
        </ProgressLabel>
        <ProgressValue $color={progressColor}>
          {Math.round(taskProgress)}%
        </ProgressValue>
      </ProgressHeader>

      {/* Enhanced Progress Bar with dual-layer visualization */}
      <ProgressBar>
        <ProgressFill
          $progress={taskProgress}
          $color={progressColor}
          $isActive={isTimerRunning && sessionType === 'POMODORO'}
        />
        <CurrentSessionOverlay
          $progress={taskProgress * (currentSessionContribution > 0 ? 1.2 : 0)}
          $isVisible={isTimerRunning && sessionType === 'POMODORO'}
        />
      </ProgressBar>

      {/* Enhanced Progress Details */}
      <ProgressDetails>
        <div>
          {currentSessionContribution > 0 ? (
            <div>
              <span style={{ color: progressColor, fontWeight: '700', fontSize: '15px' }}>
                {effectiveCompletedPomodoros.toFixed(1)} / {task.estimatedPomodoros} 🍅
              </span>
              {isTimerRunning && sessionType === 'POMODORO' && (
                <div style={{
                  fontSize: '12px',
                  color: '#3B82F6',
                  marginTop: '2px',
                  fontWeight: '500'
                }}>
                  Session: {getSessionTimeRemaining(sessionProgress)} remaining
                </div>
              )}
            </div>
          ) : (
            <span style={{ fontSize: '14px' }}>
              {task.completedPomodoros} / {task.estimatedPomodoros} 🍅
            </span>
          )}
        </div>

        {showTimeEstimate && remainingPomodoros > 0 && (
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '13px', color: '#6B7280' }}>
              ⏱️ {formatTimeRemaining(estimatedMinutes)} left
            </div>
            <div style={{
              fontSize: '11px',
              color: '#9CA3AF',
              marginTop: '1px'
            }}>
              {getCompletionETA()}
            </div>
          </div>
        )}
      </ProgressDetails>

      {/* Enhanced Status with motivational message */}
      <ProgressDetails>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <CompletionIndicator $status={completionStatus.status}>
            <span style={{ fontSize: '16px', marginRight: '4px' }}>
              {completionStatus.icon}
            </span>
            <span style={{ fontWeight: '600' }}>
              {completionStatus.text}
            </span>
          </CompletionIndicator>
          <span style={{
            fontSize: '12px',
            color: '#6B7280',
            fontStyle: 'italic'
          }}>
            {completionStatus.message}
          </span>
        </div>
      </ProgressDetails>
    </ProgressContainer>
  );
};

export type { TaskProgressProps };