import React from 'react';
import styled from 'styled-components';
import { Task } from '../../../../../types';

interface TaskProgressProps {
  task?: Task;
  sessionCount?: number;
  totalSessions?: number;
  showTimeEstimate?: boolean;
  className?: string;
}

const ProgressContainer = styled.div`
  background: rgba(255, 255, 255, 0.7);
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  border: 1px solid rgba(127, 168, 112, 0.1);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.sm};
    border-radius: ${({ theme }) => theme.borderRadius.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 12px;
    border-radius: 16px;
  }
`;

const ProgressHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
`;

const ProgressLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: #8B7D7B;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 13px;
  }
`;

const ProgressValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #2C3E50;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 13px;
  }
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 6px;
  background: rgba(127, 168, 112, 0.1);
  border-radius: 3px;
  overflow: hidden;
  position: relative;

  ${({ theme }) => theme.mediaQueries.tablet} {
    height: 8px;
    border-radius: 4px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    height: 10px;
    border-radius: 5px;
  }
`;

const ProgressFill = styled.div<{ $progress: number }>`
  height: 100%;
  background: linear-gradient(90deg, #7FA870 0%, #8FBC8F 100%);
  border-radius: inherit;
  width: ${({ $progress }) => $progress}%;
  transition: width 0.5s ease-in-out;
  position: relative;

  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmer 2s infinite;
  }

  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
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

export const TaskProgress: React.FC<TaskProgressProps> = ({
  task,
  sessionCount = 0,
  totalSessions = 4,
  showTimeEstimate = true,
  className,
}) => {
  if (!task) {
    return null;
  }

  const taskProgress = (task.completedPomodoros / task.estimatedPomodoros) * 100;
  const remainingPomodoros = task.estimatedPomodoros - task.completedPomodoros;
  const estimatedMinutes = remainingPomodoros * 25; // Assuming 25-min pomodoros

  // Determine completion status
  const getCompletionStatus = () => {
    if (taskProgress === 100) return { status: 'ahead' as const, text: 'Completed', icon: '✅' };
    if (taskProgress >= 75) return { status: 'ahead' as const, text: 'Almost done', icon: '🎯' };
    if (taskProgress >= 50) return { status: 'on-track' as const, text: 'On track', icon: '📊' };
    if (taskProgress >= 25) return { status: 'on-track' as const, text: 'In progress', icon: '🔄' };
    return { status: 'behind' as const, text: 'Just started', icon: '🌱' };
  };

  const completionStatus = getCompletionStatus();

  return (
    <ProgressContainer className={className}>
      <ProgressHeader>
        <ProgressLabel>Task Progress</ProgressLabel>
        <ProgressValue>{Math.round(taskProgress)}%</ProgressValue>
      </ProgressHeader>

      <ProgressBar>
        <ProgressFill $progress={taskProgress} />
      </ProgressBar>

      <ProgressDetails>
        <div>
          {task.completedPomodoros} / {task.estimatedPomodoros} pomodoros
        </div>

        {showTimeEstimate && remainingPomodoros > 0 && (
          <TimeEstimate>
            ⏱️ ~{estimatedMinutes}min left
          </TimeEstimate>
        )}
      </ProgressDetails>

      <ProgressDetails>
        <CompletionIndicator $status={completionStatus.status}>
          {completionStatus.icon} {completionStatus.text}
        </CompletionIndicator>
      </ProgressDetails>
    </ProgressContainer>
  );
};

export type { TaskProgressProps };