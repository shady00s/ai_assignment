import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import { HydrationTrackerProps } from '../../../../../types';
import { Card } from '../../../atoms';
import { WaterGlass } from './WaterGlass';
import { ProgressRing } from './ProgressRing';
import { GoalSettings } from './GoalSettings';

const HydrationContainer = styled(Card)<{ compact?: boolean }>`
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

  .hydration-icon {
    font-size: 24px;
    animation: wave 2s ease-in-out infinite;
  }

  @keyframes wave {
    0%, 100% { transform: rotate(0deg); }
    25% { transform: rotate(-5deg); }
    75% { transform: rotate(5deg); }
  }
`;

const MainContent = styled.div<{ compact?: boolean }>`
  display: ${({ compact }) => compact ? 'block' : 'grid'};
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing.lg};
  align-items: center;
`;

const WaterGlassesSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
`;

const WaterGlassesGrid = styled.div<{ compact?: boolean }>`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: ${({ theme }) => theme.spacing.sm};
  margin-top: ${({ theme }) => theme.spacing.md};

  ${({ compact, theme }) => compact && `
    grid-template-columns: repeat(2, 1fr);
    gap: ${theme.spacing.sm};
  `}
`;

const ProgressSection = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.md};
`;

const StatsDisplay = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.sm};
  text-align: center;
`;

const StatValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.primary};
`;

const StatLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.neutral[400]};
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

const ActionButton = styled.button<{ variant?: 'primary' | 'secondary' | 'danger' }>`
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
          background-color: ${theme.colors.waterBlue};
          color: white;
          &:hover { background-color: ${theme.colors.waterBlue}dd; }
        `;
      case 'secondary':
        return `
          background-color: ${theme.colors.neutral[100]};
          color: ${theme.colors.neutral[600]};
          border: 1px solid ${theme.colors.neutral[200]};
          &:hover { background-color: ${theme.colors.neutral[200]}; }
        `;
      case 'danger':
        return `
          background-color: ${theme.colors.error};
          color: white;
          &:hover { background-color: ${theme.colors.error}dd; }
        `;
      default:
        return '';
    }
  }}

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const SettingsButton = styled.button`
  background: none;
  border: none;
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.xs};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s ease;

  &:hover {
    background-color: ${({ theme }) => theme.colors.neutral[100]};
  }
`;

const StreakBadge = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  background-color: ${({ theme }) => theme.colors.warning};
  color: white;
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

export const HydrationTracker: React.FC<HydrationTrackerProps> = ({
  currentGlasses,
  dailyGoal,
  glassSize = 250,
  onIncrement,
  onDecrement,
  onGoalUpdate,
  isLoading,
  compact = false,
}) => {
  const [showGoalSettings, setShowGoalSettings] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);

  const progressPercentage = Math.min(100, Math.round((currentGlasses / dailyGoal) * 100));
  const totalMl = currentGlasses * glassSize;
  const goalMl = dailyGoal * glassSize;
  const remainingGlasses = Math.max(0, dailyGoal - currentGlasses);

  const handleIncrement = useCallback(() => {
    if (isLoading || isAnimating) return;

    setIsAnimating(true);
    onIncrement();

    setTimeout(() => setIsAnimating(false), 300);
  }, [isLoading, isAnimating, onIncrement]);

  const handleDecrement = useCallback(() => {
    if (isLoading || currentGlasses === 0 || isAnimating) return;

    setIsAnimating(true);
    onDecrement();

    setTimeout(() => setIsAnimating(false), 300);
  }, [isLoading, currentGlasses, isAnimating, onDecrement]);

  const handleGoalUpdate = useCallback((newGoal: number) => {
    if (isLoading || newGoal < 1 || newGoal > 20) return;

    onGoalUpdate(newGoal);
    setShowGoalSettings(false);
  }, [isLoading, onGoalUpdate]);

  const isGoalReached = currentGlasses >= dailyGoal;

  return (
    <>
      <HydrationContainer compact={compact}>
        <CardHeader>
          <h3>
            <span className="hydration-icon">💧</span>
            Hydration Tracker
          </h3>
          <SettingsButton
            onClick={() => setShowGoalSettings(true)}
            aria-label="Hydration settings"
          >
            ⚙️
          </SettingsButton>
        </CardHeader>

        <MainContent compact={compact}>
          <WaterGlassesSection>
            <WaterGlassesGrid compact={compact}>
              {Array.from({ length: Math.min(compact ? 6 : dailyGoal) }, (_, index) => (
                <WaterGlass
                  key={index}
                  filled={index < currentGlasses}
                  onClick={index === currentGlasses && currentGlasses < dailyGoal ? handleIncrement : undefined}
                  onRightClick={index === currentGlasses - 1 ? (e) => {
                    e.preventDefault();
                    handleDecrement();
                  } : undefined}
                  disabled={isLoading}
                  animationDelay={index * 50}
                />
              ))}
            </WaterGlassesGrid>

            {dailyGoal > 6 && !compact && (
              <div style={{ textAlign: 'center', marginTop: 8 }}>
                <span style={{ fontSize: '12px', color: '#666' }}>
                  +{dailyGoal - 6} more glasses
                </span>
              </div>
            )}
          </WaterGlassesSection>

          <ProgressSection>
            <ProgressRing
              percentage={progressPercentage}
              size={compact ? 80 : 120}
              strokeWidth={6}
              color="#6B8E9F"
              isComplete={isGoalReached}
            >
              <StatsDisplay>
                <StatValue>{progressPercentage}%</StatValue>
                <StatLabel>Daily Goal</StatLabel>
              </StatsDisplay>
            </ProgressRing>

            <StatsDisplay>
              <StatValue>{currentGlasses}/{dailyGoal}</StatValue>
              <StatLabel>Glasses</StatLabel>
            </StatsDisplay>
          </ProgressSection>
        </MainContent>

        <StatsDisplay style={{ marginTop: 16 }}>
          <StatValue>{totalMl}ml</StatValue>
          <StatLabel>of {goalMl}ml goal</StatLabel>
        </StatsDisplay>

        <QuickActions compact={compact}>
          <ActionButton
            variant="primary"
            onClick={handleIncrement}
            disabled={isLoading || currentGlasses >= dailyGoal || isAnimating}
          >
            +1 Glass
          </ActionButton>
          <ActionButton
            variant="secondary"
            onClick={handleDecrement}
            disabled={isLoading || currentGlasses === 0 || isAnimating}
          >
            -1 Glass
          </ActionButton>
        </QuickActions>

        {isGoalReached && (
          <div style={{ textAlign: 'center', marginTop: 16 }}>
            <StreakBadge>
              🔥 Goal Reached! Great job!
            </StreakBadge>
          </div>
        )}

        {remainingGlasses > 0 && !compact && (
          <div style={{ textAlign: 'center', marginTop: 8 }}>
            <span style={{ fontSize: '14px', color: '#666' }}>
              {remainingGlasses} more glasses to reach your goal
            </span>
          </div>
        )}
      </HydrationContainer>

      {showGoalSettings && (
        <GoalSettings
          currentGoal={dailyGoal}
          onSave={handleGoalUpdate}
          onClose={() => setShowGoalSettings(false)}
        />
      )}
    </>
  );
};