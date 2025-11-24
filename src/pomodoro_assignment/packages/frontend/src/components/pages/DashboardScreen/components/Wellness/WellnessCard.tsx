import React from 'react';
import styled from 'styled-components';
 import {
  formatMoodRating,
  formatStressLevel,
  formatEnergyLevel,
  calculatePercentage
} from '../../utils/dataFormatters';
import { Card } from '@/components/atoms';
 
interface WellnessCardProps {
  mindfulnessMinutes: number;
  hydrationGoal: number;
  hydrationCurrent: number;
  movementGoal: number;
  movementCurrent: number;
  moodRating: number;
  stressLevel: number;
  energyLevel: number;
}

const WellnessContainer = styled(Card)`
  padding: ${({ theme }) => theme.spacing.lg};
  height: fit-content;
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
  }
`;

const WellnessGrid = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const WellnessMetric = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const MetricHeader = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.sm};

  .metric-icon {
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
  }

  .metric-label {
    color: ${({ theme }) => theme.colors.neutral[500]};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  }
`;

const ProgressBarContainer = styled.div`
  width: 100%;
  height: 8px;
  background-color: ${({ theme }) => theme.colors.neutral[200]};
  border-radius: 4px;
  overflow: hidden;
`;

const ProgressBar = styled.div<{ percentage: number; color: string }>`
  height: 100%;
  width: ${({ percentage }) => percentage}%;
  background-color: ${({ color }) => color};
  border-radius: 4px;
  transition: width 0.3s ease-in-out;
`;

const MetricText = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 4px;
`;

const MetricCurrent = styled.span`
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const MetricGoal = styled.span`
  color: ${({ theme }) => theme.colors.neutral[400]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
`;

const MoodEnergyGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing.md};
  padding-top: ${({ theme }) => theme.spacing.md};
  border-top: 1px solid ${({ theme }) => theme.colors.neutral[200]};
`;

const MoodIndicator = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  text-align: center;
`;

const MoodEmoji = styled.div`
  font-size: 32px;
  line-height: 1;
`;

const MoodLabel = styled.div`
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const getMetricColor = (type: 'mindfulness' | 'hydration' | 'movement'): string => {
  switch (type) {
    case 'mindfulness': return '#7FA870'; // Sage green
    case 'hydration': return '#6B8E9F';   // Water blue
    case 'movement': return '#E67E50';   // Sunrise orange
    default: return '#7A8B7F';           // Moss green
  }
};

export const WellnessCard: React.FC<WellnessCardProps> = ({
  mindfulnessMinutes,
  hydrationGoal,
  hydrationCurrent,
  movementGoal,
  movementCurrent,
  moodRating,
  stressLevel,
  energyLevel
}) => {
  const mindfulnessPercentage = calculatePercentage(mindfulnessMinutes, 30); // 30 min goal
  const hydrationPercentage = calculatePercentage(hydrationCurrent, hydrationGoal);
  const movementPercentage = calculatePercentage(movementCurrent, movementGoal);

  return (
    <WellnessContainer>
      <CardHeader>
        <h3>Wellness Metrics</h3>
      </CardHeader>

      <WellnessGrid>
        {/* Mindfulness */}
        <WellnessMetric>
          <MetricHeader>
            <span className="metric-icon">🧘</span>
            <span className="metric-label">Mindfulness</span>
          </MetricHeader>
          <ProgressBarContainer>
            <ProgressBar
              percentage={mindfulnessPercentage}
              color={getMetricColor('mindfulness')}
            />
          </ProgressBarContainer>
          <MetricText>
            <MetricCurrent>{mindfulnessMinutes} min</MetricCurrent>
            <MetricGoal>Goal: 30 min</MetricGoal>
          </MetricText>
        </WellnessMetric>

        {/* Hydration */}
        <WellnessMetric>
          <MetricHeader>
            <span className="metric-icon">💧</span>
            <span className="metric-label">Hydration</span>
          </MetricHeader>
          <ProgressBarContainer>
            <ProgressBar
              percentage={hydrationPercentage}
              color={getMetricColor('hydration')}
            />
          </ProgressBarContainer>
          <MetricText>
            <MetricCurrent>{hydrationCurrent} glasses</MetricCurrent>
            <MetricGoal>Goal: {hydrationGoal} glasses</MetricGoal>
          </MetricText>
        </WellnessMetric>

        {/* Movement */}
        <WellnessMetric>
          <MetricHeader>
            <span className="metric-icon">🚶</span>
            <span className="metric-label">Movement</span>
          </MetricHeader>
          <ProgressBarContainer>
            <ProgressBar
              percentage={movementPercentage}
              color={getMetricColor('movement')}
            />
          </ProgressBarContainer>
          <MetricText>
            <MetricCurrent>{movementCurrent} breaks</MetricCurrent>
            <MetricGoal>Goal: {movementGoal} breaks</MetricGoal>
          </MetricText>
        </WellnessMetric>
      </WellnessGrid>

      {/* Mood, Stress, Energy */}
      <MoodEnergyGrid>
        <MoodIndicator>
          <MoodEmoji>{formatMoodRating(moodRating)}</MoodEmoji>
          <MoodLabel>Mood</MoodLabel>
        </MoodIndicator>

        <MoodIndicator>
          <MoodEmoji>{formatStressLevel(stressLevel)}</MoodEmoji>
          <MoodLabel>Stress</MoodLabel>
        </MoodIndicator>

        <MoodIndicator>
          <MoodEmoji>{formatEnergyLevel(energyLevel)}</MoodEmoji>
          <MoodLabel>Energy</MoodLabel>
        </MoodIndicator>
      </MoodEnergyGrid>
    </WellnessContainer>
  );
};