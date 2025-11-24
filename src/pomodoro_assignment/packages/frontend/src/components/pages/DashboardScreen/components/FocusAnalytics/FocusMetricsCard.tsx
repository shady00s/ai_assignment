import React from 'react';
import styled from 'styled-components';
 import {
  formatMinutes,
  calculatePercentage,
  getTrendColor,
  getTrendIcon,
  getStreakMessage,
  type TrendType
} from '../../utils/dataFormatters';
import { Card } from '@/components/atoms';

interface FocusMetricsCardProps {
  dailyFocusTime: number;
  weeklyFocusTime: number;
  monthlyFocusTime: number;
  averageSessionLength: number;
  completionRate: number;
  focusTrend: TrendType;
  streak: number;
  dailyGoal: number;
}

const FocusMetricsContainer = styled(Card)`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
  padding: ${({ theme }) => theme.spacing.lg};
`;

const CardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.md};

  h3 {
    color: ${({ theme }) => theme.colors.neutral[500]};
    font-size: ${({ theme }) => theme.typography.fontSize.xl};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    margin: 0;
  }
`;

const ProgressSection = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
`;

const ProgressRingContainer = styled.div`
  position: relative;
  width: 120px;
  height: 120px;
  margin-bottom: ${({ theme }) => theme.spacing.md};

  ${({ theme }) => theme.mediaQueries.tablet} {
    width: 140px;
    height: 140px;
  }
`;

const ProgressRing = styled.svg<{ size: number; strokeWidth: number }>`
  transform: rotate(-90deg);
  width: ${({ size }) => size}px;
  height: ${({ size }) => size}px;
`;

const ProgressRingBackground = styled.circle<{ strokeWidth: number }>`
  fill: none;
  stroke: ${({ theme }) => theme.colors.neutral[200]};
  stroke-width: ${({ strokeWidth }) => strokeWidth}px;
`;

const ProgressRingFill = styled.circle<{
  strokeWidth: number;
  progress: number;
  color: string
}>`
  fill: none;
  stroke: ${({ color }) => color};
  stroke-width: ${({ strokeWidth }) => strokeWidth}px;
  stroke-linecap: round;
  stroke-dasharray: ${({ progress }) => {
    const radius = (120 - 16) / 2; // Adjust based on size and strokeWidth
    const circumference = 2 * Math.PI * radius;
    return `${circumference * progress / 100} ${circumference}`;
  }};
  transition: stroke-dasharray 0.5s ease-in-out;
`;

const ProgressText = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
`;

const TimeDisplay = styled.div`
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
`;

const GoalDisplay = styled.div`
  color: ${({ theme }) => theme.colors.neutral[400]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  margin-top: 2px;
`;

const StreakSection = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.md} 0;
  border-top: 1px solid ${({ theme }) => theme.colors.neutral[200]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.neutral[200]};
`;

const StreakDisplay = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.primary.main};
`;

const StreakMessage = styled.div`
  color: ${({ theme }) => theme.colors.neutral[400]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  margin-top: ${({ theme }) => theme.spacing.xs};
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${({ theme }) => theme.spacing.md};
`;

const StatItem = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.sm};
  background-color: ${({ theme }) => theme.colors.neutral[50]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
`;

const StatLabel = styled.div`
  color: ${({ theme }) => theme.colors.neutral[400]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  margin-bottom: 2px;
`;

const StatValue = styled.div`
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
`;

const TrendIndicator = styled.div<{ trend: TrendType }>`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ trend }) => getTrendColor(trend)};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

export const FocusMetricsCard: React.FC<FocusMetricsCardProps> = ({
  dailyFocusTime,
  weeklyFocusTime,
  monthlyFocusTime,
  averageSessionLength,
  completionRate,
  focusTrend,
  streak,
  dailyGoal
}) => {
  const progressPercentage = calculatePercentage(dailyFocusTime, dailyGoal);

  return (
    <FocusMetricsContainer>
      <CardHeader>
        <h3>Today's Focus</h3>
        <TrendIndicator trend={focusTrend}>
          <span>{getTrendIcon(focusTrend)}</span>
          <span>{focusTrend.toLowerCase()}</span>
        </TrendIndicator>
      </CardHeader>

      <ProgressSection>
        <ProgressRingContainer>
          <ProgressRing
            size={120}
            strokeWidth={8}
            viewBox="0 0 120 120"
          >
            <ProgressRingBackground
              strokeWidth={8}
              cx="60"
              cy="60"
              r="52"
            />
            <ProgressRingFill
              strokeWidth={8}
              progress={progressPercentage}
              color="#7A8B7F"
              cx="60"
              cy="60"
              r="52"
            />
          </ProgressRing>
          <ProgressText>
            <TimeDisplay>{formatMinutes(dailyFocusTime)}</TimeDisplay>
            <GoalDisplay>of {formatMinutes(dailyGoal)}</GoalDisplay>
          </ProgressText>
        </ProgressRingContainer>
      </ProgressSection>

      <StreakSection>
        <StreakDisplay>
          <span>🔥</span>
          <span>{streak} day{streak !== 1 ? 's' : ''}</span>
        </StreakDisplay>
        <StreakMessage>{getStreakMessage(streak)}</StreakMessage>
      </StreakSection>

      <StatsGrid>
        <StatItem>
          <StatLabel>Weekly Focus</StatLabel>
          <StatValue>{formatMinutes(weeklyFocusTime)}</StatValue>
        </StatItem>
        <StatItem>
          <StatLabel>Avg Session</StatLabel>
          <StatValue>{averageSessionLength.toFixed(1)}m</StatValue>
        </StatItem>
        <StatItem>
          <StatLabel>Completion</StatLabel>
          <StatValue>{completionRate.toFixed(1)}%</StatValue>
        </StatItem>
        <StatItem>
          <StatLabel>Monthly Total</StatLabel>
          <StatValue>{formatMinutes(monthlyFocusTime)}</StatValue>
        </StatItem>
      </StatsGrid>
    </FocusMetricsContainer>
  );
};