import React, { useState, useCallback, useEffect } from 'react';
import styled from 'styled-components';
import { MovementTrackerProps } from '../../../../../types';
import { Card } from '../../../atoms';
import { MovementTimer } from './MovementTimer';
import { ActivityLogger } from './ActivityLogger';
import { ProgressRing } from '../HydrationTracker/ProgressRing';

const MovementContainer = styled(Card)<{ compact?: boolean }>`
  padding: ${({ theme, compact }) => compact ? theme.spacing.md : theme.spacing.lg};
  height: fit-content;
  position: relative;
`;

const CardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.lg};

  h3 {
    color: ${({ theme }) => theme.colors.neutral[500]};
    font-size: ${({ theme }) => theme.typography.fontSize.xl};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    margin: 0;
    display: flex;
    align-items: center;
    gap: ${({ theme }) => theme.spacing.sm};
  }

  .movement-icon {
    font-size: 24px;
    animation: bounce 2s ease-in-out infinite;
  }

  @keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
      transform: translateY(0);
    }
    40% {
      transform: translateY(-10px);
    }
    60% {
      transform: translateY(-5px);
    }
  }
`;

const MainContent = styled.div<{ compact?: boolean }>`
  display: ${({ compact }) => compact ? 'block' : 'grid'};
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing.lg};
  align-items: start;
`;

const TimerSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
`;

const ProgressSection = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.md};
`;

const StatsGrid = styled.div<{ compact?: boolean }>`
  display: grid;
  grid-template-columns: ${({ compact }) => compact ? '1fr' : '1fr 1fr'};
  gap: ${({ theme }) => theme.spacing.md};
  margin-top: ${({ theme }) => theme.spacing.lg};
`;

const StatCard = styled.div<{ variant?: 'primary' | 'secondary' | 'accent' }>`
  padding: ${({ theme }) => theme.spacing.md};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  ${({ variant = 'primary', theme }) => {
    switch (variant) {
      case 'primary':
        return `
          background: ${theme.colors.sunriseOrange}10;
          border: 1px solid ${theme.colors.sunriseOrange}30;
        `;
      case 'secondary':
        return `
          background: ${theme.colors.primary}10;
          border: 1px solid ${theme.colors.primary}30;
        `;
      case 'accent':
        return `
          background: ${theme.colors.sageGreen}10;
          border: 1px solid ${theme.colors.sageGreen}30;
        `;
    }
  }}

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
`;

const StatIcon = styled.div`
  font-size: 24px;
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const StatValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.neutral[700]};
  margin-bottom: 2px;
`;

const StatLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const QuickActions = styled.div<{ compact?: boolean }>`
  display: ${({ compact }) => compact ? 'flex' : 'grid'};
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing.md};
  margin-top: ${({ theme }) => theme.spacing.lg};

  ${({ compact, theme }) => compact && `
    gap: ${theme.spacing.sm};
    margin-top: ${theme.spacing.md};
  `}
`;

const ActionButton = styled.button<{ variant?: 'primary' | 'secondary' | 'success' }>`
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border: none;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.xs};

  ${({ variant = 'primary', theme }) => {
    switch (variant) {
      case 'primary':
        return `
          background-color: ${theme.colors.sunriseOrange};
          color: white;
          &:hover { background-color: ${theme.colors.sunriseOrange}dd; }
        `;
      case 'secondary':
        return `
          background-color: ${theme.colors.neutral[100]};
          color: ${theme.colors.neutral[600]};
          border: 1px solid ${theme.colors.neutral[200]};
          &:hover { background-color: ${theme.colors.neutral[200]}; }
        `;
      case 'success':
        return `
          background-color: ${theme.colors.success};
          color: white;
          &:hover { background-color: ${theme.colors.success}dd; }
        `;
    }
  }}

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ActivityHistory = styled.div`
  margin-top: ${({ theme }) => theme.spacing.lg};
  padding-top: ${({ theme }) => theme.spacing.lg};
  border-top: 1px solid ${({ theme }) => theme.colors.neutral[200]};
`;

const HistoryTitle = styled.h4`
  margin: 0 0 ${({ theme }) => theme.spacing.md} 0;
  color: ${({ theme }) => theme.colors.neutral[600]};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
`;

const ActivityList = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.sm};
  max-height: 200px;
  overflow-y: auto;
`;

const ActivityItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${({ theme }) => theme.spacing.sm};
  background: ${({ theme }) => theme.colors.neutral[50]};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};

  .activity-type {
    font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
    color: ${({ theme }) => theme.colors.neutral[700]};
  }

  .activity-duration {
    color: ${({ theme }) => theme.colors.neutral[500]};
  }

  .activity-time {
    color: ${({ theme }) => theme.colors.neutral[400]};
    font-size: ${({ theme }) => theme.typography.fontSize.xs};
  }
`;

const AchievementBadge = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.warning}, ${({ theme }) => theme.colors.warning}dd);
  color: white;
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  margin-top: ${({ theme }) => theme.spacing.md};
`;

const movementTypes = [
  { id: 'WALK', name: 'Walking', icon: '🚶', intensity: 'LOW' },
  { id: 'STRETCH', name: 'Stretching', icon: '🧘', intensity: 'LOW' },
  { id: 'EXERCISE', name: 'Exercise', icon: '💪', intensity: 'HIGH' },
  { id: 'DANCE', name: 'Dance', icon: '🕺', intensity: 'MEDIUM' },
  { id: 'YOGA', name: 'Yoga', icon: '🧘‍♀️', intensity: 'MEDIUM' },
  { id: 'SPORTS', name: 'Sports', icon: '⚽', intensity: 'HIGH' },
];

const recentActivities = [
  { type: 'Walking', duration: 5, time: '10:30 AM' },
  { type: 'Stretching', duration: 3, time: '9:15 AM' },
  { type: 'Exercise', duration: 15, time: '8:00 AM' },
];

export const MovementTracker: React.FC<MovementTrackerProps> = ({
  movementBreaks,
  movementMinutes,
  stepsCount,
  dailyGoal = 5,
  onStartBreak,
  onEndBreak,
  onLogActivity,
  isLoading,
  compact = false,
}) => {
  const [isTimerActive, setIsTimerActive] = useState(false);
  const [showActivityLogger, setShowActivityLogger] = useState(false);
  const [timerStartTime, setTimerStartTime] = useState<Date | null>(null);

  const breakPercentage = Math.min(100, Math.round((movementBreaks / dailyGoal) * 100));
  const remainingBreaks = Math.max(0, dailyGoal - movementBreaks);
  const isGoalReached = movementBreaks >= dailyGoal;

  const handleStartBreak = useCallback(() => {
    if (isLoading) return;

    setIsTimerActive(true);
    setTimerStartTime(new Date());
    onStartBreak();
  }, [isLoading, onStartBreak]);

  const handleEndBreak = useCallback((duration: number) => {
    if (isLoading) return;

    setIsTimerActive(false);
    setTimerStartTime(null);
    onEndBreak(duration);
  }, [isLoading, onEndBreak]);

  const handleQuickLog = useCallback((minutes: number) => {
    if (isLoading) return;

    onLogActivity(minutes, 'BREAK');
  }, [isLoading, onLogActivity]);

  const handleActivityLog = useCallback((duration: number, type: string, intensity: string) => {
    if (isLoading) return;

    onLogActivity(duration, type);
    setShowActivityLogger(false);
  }, [isLoading, onLogActivity]);

  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isTimerActive && timerStartTime) {
      interval = setInterval(() => {
        // This would typically update a timer display
        // For now, we'll let the MovementTimer component handle its own state
      }, 1000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isTimerActive, timerStartTime]);

  return (
    <>
      <MovementContainer compact={compact}>
        <CardHeader>
          <h3>
            <span className="movement-icon">🚶</span>
            Movement Tracker
          </h3>
        </CardHeader>

        <MainContent compact={compact}>
          <TimerSection>
            <MovementTimer
              isActive={isTimerActive}
              onStart={handleStartBreak}
              onEnd={handleEndBreak}
              disabled={isLoading}
              compact={compact}
            />

            <QuickActions compact={compact}>
              <ActionButton
                variant="success"
                onClick={() => handleQuickLog(5)}
                disabled={isLoading}
              >
                +5 min
              </ActionButton>
              <ActionButton
                variant="secondary"
                onClick={() => setShowActivityLogger(true)}
                disabled={isLoading}
              >
                Log Activity
              </ActionButton>
            </QuickActions>
          </TimerSection>

          <ProgressSection>
            <ProgressRing
              percentage={breakPercentage}
              size={compact ? 80 : 120}
              strokeWidth={6}
              color="#E67E50"
              isComplete={isGoalReached}
            />
            <StatCard>
              <StatIcon>🎯</StatIcon>
              <StatValue>{movementBreaks}/{dailyGoal}</StatValue>
              <StatLabel>Daily Breaks</StatLabel>
            </StatCard>
          </ProgressSection>
        </MainContent>

        <StatsGrid compact={compact}>
          <StatCard variant="primary">
            <StatIcon>⏱️</StatIcon>
            <StatValue>{movementMinutes}</StatValue>
            <StatLabel>Active Minutes</StatLabel>
          </StatCard>

          <StatCard variant="secondary">
            <StatIcon>👟</StatIcon>
            <StatValue>{stepsCount?.toLocaleString() || '0'}</StatValue>
            <StatLabel>Steps Today</StatLabel>
          </StatCard>
        </StatsGrid>

        {isGoalReached && (
          <AchievementBadge>
            🏆 Goal Reached! {movementBreaks} breaks completed
          </AchievementBadge>
        )}

        {remainingBreaks > 0 && !compact && (
          <div style={{ textAlign: 'center', marginTop: 8 }}>
            <span style={{ fontSize: '14px', color: '#666' }}>
              {remainingBreaks} more breaks to reach your goal
            </span>
          </div>
        )}

        {!compact && recentActivities.length > 0 && (
          <ActivityHistory>
            <HistoryTitle>Recent Activities</HistoryTitle>
            <ActivityList>
              {recentActivities.map((activity, index) => (
                <ActivityItem key={index}>
                  <div>
                    <span className="activity-type">{activity.type}</span>
                    <span className="activity-duration"> • {activity.duration} min</span>
                  </div>
                  <span className="activity-time">{activity.time}</span>
                </ActivityItem>
              ))}
            </ActivityList>
          </ActivityHistory>
        )}
      </MovementContainer>

      {showActivityLogger && (
        <ActivityLogger
          movementTypes={movementTypes}
          onLog={handleActivityLog}
          onClose={() => setShowActivityLogger(false)}
        />
      )}
    </>
  );
};