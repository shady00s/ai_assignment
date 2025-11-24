import React, { useState } from 'react';
import styled from 'styled-components';
import { MoodTrackerProps } from '../../../../../types';
import { Card } from '../../../atoms';
import { MoodSelector } from './MoodSelector';
import { CheckInModal } from './CheckInModal';

const MoodContainer = styled(Card)<{ compact?: boolean }>`
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

  .mood-icon {
    font-size: 24px;
    animation: pulse 3s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% {
      transform: scale(1);
    }
    50% {
      transform: scale(1.1);
    }
  }
`;

const MoodGrid = styled.div<{ compact?: boolean }>`
  display: grid;
  grid-template-columns: ${({ compact }) => compact ? '1fr' : 'repeat(3, 1fr)'};
  gap: ${({ theme }) => theme.spacing.lg};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const MoodSection = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.md};
  text-align: center;
`;

const MoodLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.neutral[600]};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const CurrentMoodDisplay = styled.div<{ compact?: boolean }>`
  font-size: ${({ compact }) => compact ? '48px' : '64px'};
  line-height: 1;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
  cursor: pointer;
  transition: transform 0.2s ease;
  user-select: none;

  &:hover {
    transform: scale(1.1);
  }

  &:active {
    transform: scale(0.95);
  }
`;

const MoodDescription = styled.div<{ value: number; compact?: boolean }>`
  font-size: ${({ theme, compact }) => compact ? theme.typography.fontSize.sm : theme.typography.fontSize.base};
  color: ${({ theme, value }) => {
    if (value >= 4) return theme.colors.success;
    if (value >= 3) return theme.colors.warning;
    return theme.colors.error;
  };
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const MoodTrend = styled.div<{ trend: 'up' | 'down' | 'stable' }>`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme, trend }) => {
    switch (trend) {
      case 'up': return theme.colors.success;
      case 'down': return theme.colors.error;
      default: return theme.colors.neutral[500];
    }
  }};
`;

const ActionButton = styled.button`
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.primary}, ${({ theme }) => theme.colors.primary}dd);
  color: white;
  border: none;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const StreakBadge = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.warning}, ${({ theme }) => theme.colors.warning}dd);
  color: white;
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const LastCheckIn = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.neutral[400]};
  text-align: center;
  margin-top: ${({ theme }) => theme.spacing.sm};
`;

const InsightsContainer = styled.div`
  margin-top: ${({ theme }) => theme.spacing.lg};
  padding-top: ${({ theme }) => theme.spacing.lg};
  border-top: 1px solid ${({ theme }) => theme.colors.neutral[200]};
`;

const InsightsTitle = styled.h4`
  margin: 0 0 ${({ theme }) => theme.spacing.md} 0;
  color: ${({ theme }) => theme.colors.neutral[600]};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  text-align: center;
`;

const InsightsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing.md};
`;

const InsightCard = styled.div`
  padding: ${({ theme }) => theme.spacing.sm};
  background: ${({ theme }) => theme.colors.neutral[50]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  text-align: center;

  .insight-value {
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
    color: ${({ theme }) => theme.colors.primary};
    margin-bottom: 2px;
  }

  .insight-label {
    font-size: ${({ theme }) => theme.typography.fontSize.xs};
    color: ${({ theme }) => theme.colors.neutral[500]};
  }
`;

const moodEmojiMap: Record<number, string> = {
  1: '😢',
  2: '😕',
  3: '😐',
  4: '🙂',
  5: '😊',
};

const moodDescriptions: Record<number, string> = {
  1: 'Very Low',
  2: 'Low',
  3: 'Neutral',
  4: 'Good',
  5: 'Excellent',
};

const stressEmojiMap: Record<number, string> = {
  1: '😌',
  2: '😊',
  3: '😐',
  4: '😰',
  5: '😫',
};

const stressDescriptions: Record<number, string> = {
  1: 'Very Relaxed',
  2: 'Relaxed',
  3: 'Moderate',
  4: 'Stressed',
  5: 'Very Stressed',
};

const energyEmojiMap: Record<number, string> = {
  1: '😴',
  2: '🔋',
  3: '⚡',
  4: '🚀',
  5: '🔥',
};

const energyDescriptions: Record<number, string> = {
  1: 'Very Low',
  2: 'Low',
  3: 'Moderate',
  4: 'High',
  5: 'Very High',
};

export const MoodTracker: React.FC<MoodTrackerProps> = ({
  moodRating,
  stressLevel,
  energyLevel,
  onMoodUpdate,
  onStressUpdate,
  onEnergyUpdate,
  lastCheckIn,
  isLoading,
  compact = false,
}) => {
  const [showCheckInModal, setShowCheckInModal] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<'mood' | 'stress' | 'energy' | null>(null);

  const handleQuickMoodUpdate = useCallback((mood: number) => {
    if (isLoading) return;
    onMoodUpdate(mood);
  }, [isLoading, onMoodUpdate]);

  const handleQuickStressUpdate = useCallback((stress: number) => {
    if (isLoading) return;
    onStressUpdate(stress);
  }, [isLoading, onStressUpdate]);

  const handleQuickEnergyUpdate = useCallback((energy: number) => {
    if (isLoading) return;
    onEnergyUpdate(energy);
  }, [isLoading, onEnergyUpdate]);

  const handleCheckIn = (mood: number, stress: number, energy: number) => {
    onMoodUpdate(mood);
    onStressUpdate(stress);
    onEnergyUpdate(energy);
    setShowCheckInModal(false);
  };

  const formatLastCheckIn = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays > 0) {
      return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    } else if (diffHours > 0) {
      return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else {
      return 'Recently';
    }
  };

  const getOverallWellness = (): number => {
    // Invert stress (lower stress is better)
    const adjustedStress = 6 - stressLevel;
    return Math.round((moodRating + adjustedStress + energyLevel) / 3);
  };

  const getMoodTrend = (): 'up' | 'down' | 'stable' => {
    // This would typically be calculated from historical data
    // For now, we'll return 'stable' as a placeholder
    return 'stable';
  };

  const overallWellness = getOverallWellness();

  return (
    <>
      <MoodContainer compact={compact}>
        <CardHeader>
          <h3>
            <span className="mood-icon">😊</span>
            Mood & Wellness
          </h3>
          <StreakBadge>
            🔥 3 day streak
          </StreakBadge>
        </CardHeader>

        <MoodGrid compact={compact}>
          <MoodSection>
            <MoodLabel>Mood</MoodLabel>
            <CurrentMoodDisplay
              compact={compact}
              onClick={() => setSelectedMetric('mood')}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSelectedMetric('mood');
                }
              }}
              aria-label={`Current mood: ${moodDescriptions[moodRating]}. Click to update.`}
            >
              {moodEmojiMap[moodRating]}
            </CurrentMoodDisplay>
            <MoodDescription value={moodRating} compact={compact}>
              {moodDescriptions[moodRating]}
            </MoodDescription>
            <MoodTrend trend={getMoodTrend()}>
              {getMoodTrend() === 'up' ? '↗️' : getMoodTrend() === 'down' ? '↘️' : '→'} Stable
            </MoodTrend>
          </MoodSection>

          <MoodSection>
            <MoodLabel>Stress</MoodLabel>
            <CurrentMoodDisplay
              compact={compact}
              onClick={() => setSelectedMetric('stress')}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSelectedMetric('stress');
                }
              }}
              aria-label={`Current stress level: ${stressDescriptions[stressLevel]}. Click to update.`}
            >
              {stressEmojiMap[stressLevel]}
            </CurrentMoodDisplay>
            <MoodDescription value={6 - stressLevel} compact={compact}>
              {stressDescriptions[stressLevel]}
            </MoodDescription>
          </MoodSection>

          <MoodSection>
            <MoodLabel>Energy</MoodLabel>
            <CurrentMoodDisplay
              compact={compact}
              onClick={() => setSelectedMetric('energy')}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSelectedMetric('energy');
                }
              }}
              aria-label={`Current energy level: ${energyDescriptions[energyLevel]}. Click to update.`}
            >
              {energyEmojiMap[energyLevel]}
            </CurrentMoodDisplay>
            <MoodDescription value={energyLevel} compact={compact}>
              {energyDescriptions[energyLevel]}
            </MoodDescription>
          </MoodSection>
        </MoodGrid>

        {!compact && (
          <>
            <ActionButton
              onClick={() => setShowCheckInModal(true)}
              disabled={isLoading}
            >
              📝 Complete Check-In
            </ActionButton>

            <LastCheckIn>
              Last check-in: {lastCheckIn ? formatLastCheckIn(lastCheckIn) : 'Never'}
            </LastCheckIn>

            <InsightsContainer>
              <InsightsTitle>Today's Wellness</InsightsTitle>
              <InsightsGrid>
                <InsightCard>
                  <div className="insight-value">{overallWellness}/5</div>
                  <div className="insight-label">Overall Wellness</div>
                </InsightCard>
                <InsightCard>
                  <div className="insight-value">{moodRating}</div>
                  <div className="insight-label">Mood Score</div>
                </InsightCard>
              </InsightsGrid>
            </InsightsContainer>
          </>
        )}

        {compact && (
          <LastCheckIn>
            Check-in: {lastCheckIn ? formatLastCheckIn(lastCheckIn) : 'Never'}
          </LastCheckIn>
        )}
      </MoodContainer>

      {/* Individual mood selector modals */}
      {selectedMetric && (
        <MoodSelector
          metric={selectedMetric}
          currentValue={selectedMetric === 'mood' ? moodRating : selectedMetric === 'stress' ? stressLevel : energyLevel}
          onSelect={(value) => {
            if (selectedMetric === 'mood') {
              handleQuickMoodUpdate(value);
            } else if (selectedMetric === 'stress') {
              handleQuickStressUpdate(value);
            } else {
              handleQuickEnergyUpdate(value);
            }
            setSelectedMetric(null);
          }}
          onClose={() => setSelectedMetric(null)}
        />
      )}

      {/* Full check-in modal */}
      {showCheckInModal && (
        <CheckInModal
          currentMood={moodRating}
          currentStress={stressLevel}
          currentEnergy={energyLevel}
          onCheckIn={handleCheckIn}
          onClose={() => setShowCheckInModal(false)}
        />
      )}
    </>
  );
};