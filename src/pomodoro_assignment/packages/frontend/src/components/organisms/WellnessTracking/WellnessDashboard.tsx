import React, { useMemo } from 'react';
import styled from 'styled-components';
import { WellnessDashboardProps } from '../../../types';
import { Card } from '../../atoms';
import { HydrationTracker } from './components/HydrationTracker';
import { MovementTracker } from './components/MovementTracker';
import { MoodTracker } from './components/MoodTracker';
import { MeditationTimer } from './components/MeditationTimer';
import { useWellnessData } from '../../../hooks';
import { LoadingSpinner } from '../../atoms/LoadingSpinner';
import { ErrorMessage } from '../../atoms/ErrorMessage';

const DashboardContainer = styled.div<{ viewMode?: 'compact' | 'detailed' | 'analytics' }>`
  width: 100%;
  max-width: 100%;
  padding: ${({ theme, viewMode }) => {
    switch (viewMode) {
      case 'compact':
        return theme.spacing.mobile.sm;
      case 'analytics':
        return theme.spacing.desktop.lg;
      default:
        return theme.spacing.md;
    }
  }};

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme, viewMode }) => viewMode === 'analytics' ? theme.spacing.tablet.lg : theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme, viewMode }) => viewMode === 'analytics' ? theme.spacing.desktop.lg : theme.spacing.desktop.lg};
  }
`;

const DashboardHeader = styled.header`
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  text-align: center;

  h1 {
    color: ${({ theme }) => theme.colors.neutral[500]};
    font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
    margin-bottom: ${({ theme }) => theme.spacing.sm};

    ${({ theme }) => theme.mediaQueries.tablet} {
      font-size: ${({ theme }) => theme.typography.fontSize.tablet['3xl']};
    }

    ${({ theme }) => theme.mediaQueries.desktop} {
      font-size: ${({ theme }) => theme.typography.fontSize.desktop['3xl']};
    }
  }

  p {
    color: ${({ theme }) => theme.colors.neutral[400]};
    font-size: ${({ theme }) => theme.typography.fontSize.base};

    ${({ theme }) => theme.mediaQueries.tablet} {
      font-size: ${({ theme }) => theme.typography.fontSize.tablet.lg;
    }
  }
`;

const WellnessGrid = styled.div<{ viewMode?: 'compact' | 'detailed' | 'analytics' }>`
  display: grid;
  gap: ${({ theme }) => theme.spacing.md};

  ${({ viewMode, theme }) => {
    switch (viewMode) {
      case 'compact':
        return `
          grid-template-columns: 1fr;
        `;
      case 'analytics':
        return `
          grid-template-columns: repeat(2, 1fr);
        `;
      default:
        return `
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        `;
    }
  }}

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.lg};
    grid-template-columns: ${({ viewMode }) => viewMode === 'compact' ? '1fr' : 'repeat(2, 1fr)'};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: ${({ theme }) => theme.spacing.lg};
    grid-template-columns: ${({ viewMode }) => viewMode === 'analytics' ? 'repeat(2, 1fr)' : 'repeat(2, 1fr)'};
  }
`;

const FullWidthSection = styled.div<{ viewMode?: 'compact' | 'detailed' | 'analytics' }>`
  grid-column: 1 / -1;

  ${({ viewMode }) => {
    switch (viewMode) {
      case 'analytics':
        return `
          grid-column: span 2;
        `;
      default:
        return '';
    }
  }
`;

const QuickStatsContainer = styled(Card)`
  padding: ${({ theme }) => theme.spacing.md};
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const QuickStatsGrid = styled.div<{ viewMode?: 'compact' | 'detailed' | 'analytics' }>`
  display: grid;
  gap: ${({ theme }) => theme.spacing.sm};
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));

  ${({ viewMode }) => viewMode === 'compact' && `
    grid-template-columns: repeat(2, 1fr);
  `}

  ${({ theme }) => theme.mediaQueries.tablet} {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: ${({ theme }) => theme.spacing.md};
  }
`;

const StatItem = styled.div<{ variant?: 'primary' | 'secondary' | 'accent' | 'neutral' }>`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.sm};

  .stat-value {
    font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
    margin-bottom: 4px;

    ${({ variant, theme }) => {
      switch (variant) {
        case 'primary':
          return `color: ${theme.colors.waterBlue};`;
        case 'secondary':
          return `color: ${theme.colors.sunriseOrange};`;
        case 'accent':
          return `color: ${theme.colors.sageGreen};`;
        case 'neutral':
        default:
          return `color: ${theme.colors.neutral[700]};`;
      }
    }}
  }

  .stat-label {
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    color: ${({ theme }) => theme.colors.neutral[500]};
    font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  }
`;

const SummaryCard = styled(Card)`
  padding: ${({ theme }) => theme.spacing.lg};
  text-align: center;
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.primary}10, ${({ theme }) => theme.colors.success}10);
  border-left: 4px solid ${({ theme }) => theme.colors.primary};
`;

const SummaryTitle = styled.h3`
  margin: 0 0 ${({ theme }) => theme.spacing.md} 0;
  color: ${({ theme }) => theme.colors.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
`;

const SummaryScore = styled.div`
  font-size: 48px;
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.primary};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const SummaryDescription = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.neutral[600]};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  line-height: 1.4;
`;

const getTodayDate = (): string => {
  return new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });
};

export const WellnessDashboard: React.FC<WellnessDashboardProps> = ({
  date,
  viewMode = 'detailed',
  onDateChange,
  className,
}) => {
  const {
    todayWellness,
    wellnessHistory,
    analytics,
    recommendations,
    isLoading,
    hasError,
    actions,
    metrics,
    isWellnessDataAvailable,
  } = useWellnessData();

  const selectedDate = date || new Date().toISOString().split('T')[0];

  const handleHydrationIncrement = () => {
    actions.hydrate(1);
  };

  const handleHydrationDecrement = () => {
    actions.hydrate(-1);
  };

  const handleHydrationGoalUpdate = (newGoal: number) => {
    // This would call an API to update the hydration goal
    console.log('Updating hydration goal to:', newGoal);
  };

  const handleMovementStart = () => {
    console.log('Starting movement break');
  };

  const handleMovementEnd = (duration: number) => {
    actions.logMovement(duration, 'BREAK');
  };

  const handleMovementLog = (minutes: number, type: string) => {
    actions.logMovement(minutes, type);
  };

  const handleMoodUpdate = (mood: number) => {
    actions.updateMood(mood, todayWellness?.stressLevel || 3, todayWellness?.energyLevel || 3);
  };

  const handleStressUpdate = (stress: number) => {
    actions.updateMood(todayWellness?.moodRating || 3, stress, todayWellness?.energyLevel || 3);
  };

  const handleEnergyUpdate = (energy: number) => {
    actions.updateMood(todayWellness?.moodRating || 3, todayWellness?.stressLevel || 3, energy);
  };

  const handleMeditationStart = (duration: number) => {
    console.log('Starting meditation session:', duration);
  };

  const handleMeditationComplete = (duration: number, quality: number) => {
    actions.logMeditation(duration, 'MINDFULNESS', quality);
  };

  const guidedOptions = [
    {
      id: '1',
      name: '5 min Breathing',
      duration: 5,
      type: 'BREATHING',
      description: 'Quick breathing exercise for relaxation',
    },
    {
      id: '2',
      name: '10 min Mindfulness',
      duration: 10,
      type: 'MINDFULNESS',
      description: 'Basic mindfulness practice',
    },
    {
      id: '3',
      name: '15 min Guided',
      duration: 15,
      type: 'GUIDED',
      description: 'Guided meditation for focus',
    },
  ];

  if (isLoading) {
    return (
      <DashboardContainer viewMode={viewMode} className={className}>
        <LoadingSpinner size="large" centered />
      </DashboardContainer>
    );
  }

  if (hasError) {
    return (
      <DashboardContainer viewMode={viewMode} className={className}>
        <ErrorMessage
          message="Failed to load wellness data. Please try again later."
          variant="card"
        />
      </DashboardContainer>
    );
  }

  const overallWellnessScore = metrics.wellness.score;

  return (
    <DashboardContainer viewMode={viewMode} className={className}>
      <DashboardHeader>
        <h1>Wellness Dashboard</h1>
        <p>Track your health and mindfulness journey</p>
      </DashboardHeader>

      {viewMode !== 'compact' && (
        <QuickStatsContainer>
          <QuickStatsGrid viewMode={viewMode}>
            <StatItem variant="primary">
              <div className="stat-value">{metrics.hydration.current}</div>
              <div className="stat-label">💧 Water Glasses</div>
            </StatItem>

            <StatItem variant="secondary">
              <div className="stat-value">{metrics.movement.currentBreaks}</div>
              <div className="stat-label">🚶 Movement Breaks</div>
            </StatItem>

            <StatItem variant="accent">
              <div className="stat-value">{metrics.mood.current}</div>
              <div className="stat-label">😊 Mood Level</div>
            </StatItem>

            <StatItem variant="neutral">
              <div className="stat-value">{metrics.meditation.currentMinutes}</div>
              <div className="stat-label">🧘 Meditation Min</div>
            </StatItem>
          </QuickStatsGrid>
        </QuickStatsContainer>
      )}

      {viewMode !== 'compact' && (
        <FullWidthSection viewMode={viewMode}>
          <SummaryCard>
            <SummaryTitle>Today's Wellness Score</SummaryTitle>
            <SummaryScore>{overallWellnessScore}/100</SummaryScore>
            <SummaryDescription>
              {overallWellnessScore >= 80 ? "Excellent! You're taking great care of your wellness today." :
               overallWellnessScore >= 60 ? "Good job! Keep up the healthy habits." :
               overallWellnessScore >= 40 ? "You're on your way! Try to complete your wellness goals." :
               "Let's focus on improving your wellness today. Small steps make a big difference!"}
            </SummaryDescription>
          </SummaryCard>
        </FullWidthSection>
      )}

      <WellnessGrid viewMode={viewMode}>
        <HydrationTracker
          currentGlasses={metrics.hydration.current}
          dailyGoal={metrics.hydration.goal}
          glassSize={250}
          onIncrement={handleHydrationIncrement}
          onDecrement={handleHydrationDecrement}
          onGoalUpdate={handleHydrationGoalUpdate}
          isLoading={false}
          compact={viewMode === 'compact'}
        />

        <MovementTracker
          movementBreaks={metrics.movement.currentBreaks}
          movementMinutes={metrics.movement.currentMinutes}
          stepsCount={0} // This would come from wellness data
          dailyGoal={5}
          onStartBreak={handleMovementStart}
          onEndBreak={handleMovementEnd}
          onLogActivity={handleMovementLog}
          isLoading={false}
          compact={viewMode === 'compact'}
        />

        {viewMode !== 'compact' && (
          <>
            <MoodTracker
              moodRating={metrics.mood.current}
              stressLevel={6 - (todayWellness?.stressLevel || 3)} // Invert stress for display
              energyLevel={metrics.energy.current}
              onMoodUpdate={handleMoodUpdate}
              onStressUpdate={handleStressUpdate}
              onEnergyUpdate={handleEnergyUpdate}
              lastCheckIn={todayWellness?.updatedAt || ''}
              isLoading={false}
              compact={false}
            />

            <MeditationTimer
              totalMinutes={metrics.meditation.currentMinutes}
              sessionGoal={15}
              onStartSession={handleMeditationStart}
              onCompleteSession={handleMeditationComplete}
              guidedOptions={guidedOptions}
              isLoading={false}
              compact={false}
            />
          </>
        )}

        {viewMode === 'compact' && (
          <>
            <MoodTracker
              moodRating={metrics.mood.current}
              stressLevel={6 - (todayWellness?.stressLevel || 3)}
              energyLevel={metrics.energy.current}
              onMoodUpdate={handleMoodUpdate}
              onStressUpdate={handleStressUpdate}
              onEnergyUpdate={handleEnergyUpdate}
              lastCheckIn={todayWellness?.updatedAt || ''}
              isLoading={false}
              compact={true}
            />

            <MeditationTimer
              totalMinutes={metrics.meditation.currentMinutes}
              sessionGoal={15}
              onStartSession={handleMeditationStart}
              onCompleteSession={handleMeditationComplete}
              guidedOptions={guidedOptions}
              isLoading={false}
              compact={true}
            />
          </>
        )}
      </WellnessGrid>
    </DashboardContainer>
  );
};