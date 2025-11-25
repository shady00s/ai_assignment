import React from 'react';
import styled from 'styled-components';

interface TodayProgressProps {
  focusTimeMinutes: number;
  focusTimeGoal: number;
  tasksCompleted: number;
  tasksTotal: number;
  streakDays: number;
  weeklyTrend: 'up' | 'down' | 'stable';
  qualityScore: number;
  className?: string;
}

const ProgressContainer = styled.div`
  background: rgba(255, 255, 255, 0.8);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.mobile.md};
  border: 1px solid rgba(230, 126, 80, 0.1);
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 24px;
    border-radius: 20px;
  }
`;

const ProgressHeader = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.md};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #E67E50;
  margin: 0 0 ${({ theme }) => theme.spacing.mobile.md} 0;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.md};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.md};
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 18px;
    margin-bottom: 20px;
    gap: 8px;
  }
`;

const ProgressMetrics = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: ${({ theme }) => theme.spacing.mobile.md};

  ${({ theme }) => theme.mediaQueries.tablet} {
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
  }
`;

const MetricCard = styled.div<{ $color: string }>`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.mobile.md};
  background: rgba(255, 255, 255, 0.6);
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border: 1px solid ${({ $color }) => `${$color}20`};
  transition: all 0.2s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    background: rgba(255, 255, 255, 0.8);
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md};
    border-radius: ${({ theme }) => theme.borderRadius.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 16px;
    border-radius: 16px;
  }
`;

const MetricIcon = styled.div<{ $color: string }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile['2xl']};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
  color: ${({ $color }) => $color};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 28px;
    margin-bottom: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 32px;
    margin-bottom: 8px;
  }
`;

const MetricValue = styled.div<{ $color: string }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ $color }) => $color};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xs};
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.lg};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.xs};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 24px;
    margin-bottom: 4px;
  }
`;

const MetricLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #8B7D7B;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 13px;
  }
`;

const ProgressBar = styled.div<{ $color: string }>`
  width: 100%;
  height: 6px;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 3px;
  overflow: hidden;
  margin-top: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    height: 8px;
    border-radius: 4px;
    margin-top: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    height: 10px;
    border-radius: 5px;
    margin-top: 8px;
  }
`;

const ProgressFill = styled.div<{ $progress: number; $color: string }>`
  height: 100%;
  background: ${({ $color }) => $color};
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

const TrendIndicator = styled.div<{ $trend: 'up' | 'down' | 'stable' }>`
  display: inline-flex;
  align-items: center;
  gap: 2px;
  font-size: 11px;
  color: ${({ $trend }) => {
    switch ($trend) {
      case 'up': return '#7FA870';
      case 'down': return '#C85A5A';
      case 'stable': return '#F4A261';
      default: return '#F4A261';
    }
  }};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const StreakBadge = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  background: linear-gradient(135deg, #E67E50 0%, #F39C12 100%);
  color: white;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: bold;
  margin-top: 4px;
  animation: flameFlicker 2s ease-in-out infinite;

  @keyframes flameFlicker {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
  }
`;

export const TodayProgress: React.FC<TodayProgressProps> = ({
  focusTimeMinutes,
  focusTimeGoal,
  tasksCompleted,
  tasksTotal,
  streakDays,
  weeklyTrend,
  qualityScore,
  className,
}) => {
  const focusProgress = focusTimeGoal > 0 ? Math.min((focusTimeMinutes / focusTimeGoal) * 100, 100) : 0;
  const taskProgress = tasksTotal > 0 ? (tasksCompleted / tasksTotal) * 100 : 0;
  const qualityProgress = (qualityScore / 100) * 100;

  const formatTime = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
  };

  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up': return '📈';
      case 'down': return '📉';
      case 'stable': return '➡️';
      default: return '➡️';
    }
  };

  const metrics = [
    {
      icon: '⏱️',
      label: 'Focus Time',
      value: formatTime(focusTimeMinutes),
      goal: formatTime(focusTimeGoal),
      color: '#E67E50',
      progress: focusProgress,
      showProgress: true,
    },
    {
      icon: '🎯',
      label: 'Tasks Completed',
      value: `${tasksCompleted}`,
      goal: `${tasksTotal}`,
      color: '#7FA870',
      progress: taskProgress,
      showProgress: true,
    },
    {
      icon: '🏆',
      label: 'Quality Score',
      value: `${qualityScore}%`,
      goal: '100%',
      color: '#9B59B6',
      progress: qualityProgress,
      showProgress: true,
    },
    {
      icon: '🔥',
      label: 'Current Streak',
      value: `${streakDays} days`,
      goal: '',
      color: '#E67E50',
      progress: 100,
      showProgress: false,
      special: true,
    },
  ];

  return (
    <ProgressContainer className={className}>
      <ProgressHeader>
        📊 Today's Progress
        <TrendIndicator $trend={weeklyTrend}>
          {getTrendIcon(weeklyTrend)} vs last week
        </TrendIndicator>
      </ProgressHeader>

      <ProgressMetrics>
        {metrics.map((metric, index) => (
          <MetricCard key={index} $color={metric.color}>
            <MetricIcon $color={metric.color}>
              {metric.icon}
            </MetricIcon>
            <MetricValue $color={metric.color}>
              {metric.value}
            </MetricValue>
            <MetricLabel>{metric.label}</MetricLabel>

            {metric.showProgress && (
              <ProgressBar $color={metric.color}>
                <ProgressFill $progress={metric.progress} $color={metric.color} />
              </ProgressBar>
            )}

            {metric.special && streakDays > 0 && (
              <StreakBadge>
                🔥 {streakDays} day streak!
              </StreakBadge>
            )}
          </MetricCard>
        ))}
      </ProgressMetrics>
    </ProgressContainer>
  );
};

export type { TodayProgressProps };