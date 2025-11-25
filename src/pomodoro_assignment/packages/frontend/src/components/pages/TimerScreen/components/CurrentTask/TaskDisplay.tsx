import { Task } from '@/types';
import React from 'react';
import styled from 'styled-components';

interface TaskDisplayProps {
  currentTask?: Task;
  sessionCount?: number;
  totalSessions?: number;
  energyLevel?: 'LOW' | 'MEDIUM' | 'HIGH';
  isTimerRunning?: boolean;
  onTaskSelect?: () => void;
  className?: string;
}

const TaskDisplayContainer = styled.div`
  background: rgba(255, 255, 255, 0.8);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.mobile.md};
  border: 1px solid rgba(127, 168, 112, 0.1);
  backdrop-filter: blur(10px);
  position: relative;
  overflow: hidden;

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 20px;
    border-radius: 20px;
  }
`;

const TaskHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
`;

const TaskIcon = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.xl};
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 24px;
    gap: 8px;
  }
`;

const TaskTitle = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.md};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #2C3E50;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  line-height: 1.3;

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 16px;
  }
`;

const NoTaskText = styled.div`
  color: #A8968E;
  font-style: italic;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 14px;
  }
`;

const TaskProgress = styled.div`
  margin-top: ${({ theme }) => theme.spacing.mobile.sm};
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: rgba(127, 168, 112, 0.1);
  border-radius: 4px;
  overflow: hidden;
  position: relative;

  ${({ theme }) => theme.mediaQueries.tablet} {
    height: 10px;
    border-radius: 5px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    height: 12px;
    border-radius: 6px;
  }
`;

const ProgressFill = styled.div<{ $progress: number; $animated?: boolean }>`
  height: 100%;
  background: linear-gradient(90deg, #7FA870 0%, #8FBC8F 100%);
  border-radius: inherit;
  width: ${({ $progress }) => $progress}%;
  transition: width 0.5s ease-in-out;
  position: relative;

  ${({ $animated }) =>
    $animated &&
    `
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
  `}
`;

const TaskDetails = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: ${({ theme }) => theme.spacing.mobile.sm};
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing.mobile.xs};

  ${({ theme }) => theme.mediaQueries.tablet} {
    margin-top: ${({ theme }) => theme.spacing.tablet.sm};
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    margin-top: 12px;
    gap: 12px;
  }
`;

const SessionInfo = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #8B7D7B;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.xs};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 4px;
    font-size: 13px;
  }
`;

const EnergyIndicator = styled.div<{ $level: 'LOW' | 'MEDIUM' | 'HIGH' }>`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  padding: ${({ theme }) => theme.spacing.mobile.xs} ${({ theme }) => theme.spacing.mobile.sm};
  border-radius: 12px;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  background: ${({ $level }) => {
    switch ($level) {
      case 'HIGH': return 'rgba(127, 168, 112, 0.1)';
      case 'MEDIUM': return 'rgba(244, 162, 97, 0.1)';
      case 'LOW': return 'rgba(200, 90, 90, 0.1)';
      default: return 'rgba(127, 168, 112, 0.1)';
    }
  }};
  color: ${({ $level }) => {
    switch ($level) {
      case 'HIGH': return '#7FA870';
      case 'MEDIUM': return '#F4A261';
      case 'LOW': return '#C85A5A';
      default: return '#7FA870';
    }
  }};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.xs};
    padding: ${({ theme }) => theme.spacing.tablet.xs} ${({ theme }) => theme.spacing.tablet.sm};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
    border-radius: 14px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 4px;
    padding: 4px 8px;
    font-size: 12px;
    border-radius: 16px;
  }
`;

const TaskAction = styled.button`
  background: none;
  border: 1px solid rgba(127, 168, 112, 0.2);
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.mobile.xs} ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #7FA870;
  cursor: pointer;
  transition: all 0.2s ease;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};

  &:hover {
    background: rgba(127, 168, 112, 0.1);
    transform: translateY(-1px);
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.xs} ${({ theme }) => theme.spacing.tablet.sm};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
    border-radius: ${({ theme }) => theme.borderRadius.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 6px 12px;
    font-size: 12px;
    border-radius: 12px;
  }
`;

const getTaskIcon = (priority: string) => {
  switch (priority) {
    case 'URGENT': return '🚨';
    case 'HIGH': return '🔴';
    case 'MEDIUM': return '🟡';
    case 'LOW': return '🟢';
    default: return '📋';
  }
};

const getEnergyIcon = (level: 'LOW' | 'MEDIUM' | 'HIGH') => {
  switch (level) {
    case 'HIGH': return '⚡';
    case 'MEDIUM': return '🔋';
    case 'LOW': return '🪫';
    default: return '⚡';
  }
};

const getEnergyText = (level: 'LOW' | 'MEDIUM' | 'HIGH') => {
  switch (level) {
    case 'HIGH': return 'High Energy';
    case 'MEDIUM': return 'Medium Energy';
    case 'LOW': return 'Low Energy';
    default: return 'High Energy';
  }
};

export const TaskDisplay: React.FC<TaskDisplayProps> = ({
  currentTask,
  sessionCount = 0,
  totalSessions = 4,
  energyLevel = 'HIGH',
  isTimerRunning = false,
  onTaskSelect,
  className,
}) => {
  const taskProgress = currentTask ? (currentTask.completedPomodoros / currentTask.estimatedPomodoros) * 100 : 0;
  const sessionProgress = totalSessions > 0 ? (sessionCount / totalSessions) * 100 : 0;

  return (
    <TaskDisplayContainer className={className}>
      <TaskHeader>
        <TaskIcon>
          {currentTask ? getTaskIcon(currentTask.priority) : '🎧'}
          {currentTask ? (
            <TaskTitle>{currentTask.title}</TaskTitle>
          ) : (
            <NoTaskText>No task selected</NoTaskText>
          )}
        </TaskIcon>

        {onTaskSelect && (
          <TaskAction onClick={onTaskSelect}>
            {currentTask ? 'Change' : 'Select Task'}
          </TaskAction>
        )}
      </TaskHeader>

      {currentTask && (
        <TaskProgress>
          <ProgressBar>
            <ProgressFill $progress={taskProgress} $animated={isTimerRunning} />
          </ProgressBar>
        </TaskProgress>
      )}

      <TaskDetails>
        <SessionInfo>
          ⏱️ Session {sessionCount} of {totalSessions}
        </SessionInfo>

        <EnergyIndicator $level={energyLevel}>
          {getEnergyIcon(energyLevel)} {getEnergyText(energyLevel)}
        </EnergyIndicator>
      </TaskDetails>
    </TaskDisplayContainer>
  );
};

export type { TaskDisplayProps };