import React from 'react';
import styled from 'styled-components';
import { WellnessEntry } from '../../../../../types';

interface WellnessMetricsProps {
  wellnessData?: WellnessEntry;
  onIncrementHydration?: () => void;
  onStartMovement?: () => void;
  onUpdateMood?: (mood: number) => void;
  onStartMeditation?: () => void;
  compact?: boolean;
  className?: string;
}

const WellnessContainer = styled.div<{ $compact: boolean }>`
  background: rgba(255, 255, 255, 0.8);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ $compact, theme }) => $compact ? theme.spacing.mobile.md : theme.spacing.mobile.lg};
  border: 1px solid rgba(127, 168, 112, 0.1);
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ $compact, theme }) => $compact ? theme.spacing.tablet.md : theme.spacing.tablet.lg};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ $compact }) => $compact ? '20px' : '24px'};
    border-radius: 20px;
  }
`;

const WellnessTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.md};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #2C3E50;
  margin: 0 0 ${({ theme }) => theme.spacing.mobile.md} 0;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
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

const MetricsGrid = styled.div<{ $compact: boolean }>`
  display: grid;
  grid-template-columns: ${({ $compact }) => $compact ? 'repeat(2, 1fr)' : 'repeat(auto-fit, minmax(140px, 1fr))'};
  gap: ${({ theme }) => theme.spacing.mobile.md};

  ${({ theme }) => theme.mediaQueries.tablet} {
    grid-template-columns: ${({ $compact }) => $compact ? 'repeat(2, 1fr)' : 'repeat(auto-fit, minmax(160px, 1fr))'};
    gap: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    grid-template-columns: ${({ $compact }) => $compact ? 'repeat(2, 1fr)' : 'repeat(auto-fit, minmax(180px, 1fr))'};
    gap: 16px;
  }
`;

const MetricCard = styled.div<{ $interactive?: boolean }>`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.mobile.md};
  background: rgba(255, 255, 255, 0.6);
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border: 1px solid rgba(127, 168, 112, 0.1);
  transition: all 0.2s ease;
  cursor: ${({ $interactive }) => $interactive ? 'pointer' : 'default'};
  position: relative;

  &:hover {
    transform: ${({ $interactive }) => $interactive ? 'translateY(-2px)' : 'none'};
    box-shadow: ${({ $interactive }) => $interactive ? '0 4px 12px rgba(127, 168, 112, 0.15)' : 'none'};
    background: ${({ $interactive }) => $interactive ? 'rgba(255, 255, 255, 0.8)' : 'rgba(255, 255, 255, 0.6)'};
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
  opacity: 0.8;

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 28px;
    margin-bottom: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 32px;
    margin-bottom: 8px;
  }
`;

const MetricLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: #8B7D7B;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xs};
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  text-transform: uppercase;
  letter-spacing: 0.5px;

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.xs};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.xs};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 11px;
    margin-bottom: 4px;
  }
`;

const MetricValue = styled.div<{ $color: string }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ $color }) => $color};
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xs};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.lg};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.xs};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 18px;
    margin-bottom: 4px;
  }
`;

const ProgressBar = styled.div<{ $color: string }>`
  width: 100%;
  height: 4px;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 2px;
  overflow: hidden;
  position: relative;

  ${({ theme }) => theme.mediaQueries.tablet} {
    height: 6px;
    border-radius: 3px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    height: 8px;
    border-radius: 4px;
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

const MetricAction = styled.button`
  background: none;
  border: 1px solid rgba(127, 168, 112, 0.2);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 10px;
  color: #7FA870;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-top: 4px;

  &:hover {
    background: rgba(127, 168, 112, 0.1);
    transform: translateY(-1px);
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: 6px 12px;
    font-size: 11px;
    border-radius: 8px;
    margin-top: 6px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 8px 16px;
    font-size: 12px;
    border-radius: 12px;
    margin-top: 8px;
  }
`;

interface MetricData {
  icon: string;
  label: string;
  value: string;
  current: number;
  goal: number;
  color: string;
  interactive: boolean;
  action?: string;
  onClick?: () => void;
}

export const WellnessMetrics: React.FC<WellnessMetricsProps> = ({
  wellnessData,
  onIncrementHydration,
  onStartMovement,
  onUpdateMood,
  onStartMeditation,
  compact = false,
  className,
}) => {
  // Default wellness data if not provided
  const defaultWellnessData: WellnessEntry = {
    id: '',
    userId: '',
    date: new Date().toISOString(),
    hydrationGlasses: 6,
    hydrationGoal: 8,
    movementBreaks: 3,
    movementMinutes: 15,
    meditationMinutes: 15,
    breathingExercises: 2,
    mindfulnessSessions: 1,
    moodRating: 4,
    stressLevel: 2,
    energyLevel: 4,
    postureChecks: 8,
    eyeRestBreaks: 12,
    createdAt: '',
    updatedAt: '',
  };

  const data = wellnessData || defaultWellnessData;

  const metrics: MetricData[] = [
    {
      icon: '💧',
      label: 'Hydration',
      value: `${data.hydrationGlasses}/${data.hydrationGoal}`,
      current: data.hydrationGlasses,
      goal: data.hydrationGoal,
      color: '#4A90E2',
      interactive: true,
      action: '+1 Glass',
      onClick: onIncrementHydration,
    },
    {
      icon: '🧘',
      label: 'Mindfulness',
      value: `${data.meditationMinutes}min`,
      current: data.meditationMinutes,
      goal: 30,
      color: '#9B59B6',
      interactive: true,
      action: 'Meditate',
      onClick: onStartMeditation,
    },
    {
      icon: '🚶',
      label: 'Movement',
      value: `${data.movementBreaks} breaks`,
      current: data.movementBreaks,
      goal: 8,
      color: '#E67E22',
      interactive: true,
      action: 'Move Now',
      onClick: onStartMovement,
    },
    {
      icon: '😊',
      label: 'Mood',
      value: '⭐'.repeat(data.moodRating),
      current: data.moodRating,
      goal: 5,
      color: '#F39C12',
      interactive: true,
      action: 'Update',
      onClick: () => onUpdateMood?.(data.moodRating),
    },
  ];

  return (
    <WellnessContainer $compact={compact} className={className}>
      <WellnessTitle>
        🌿 Wellness Metrics
      </WellnessTitle>

      <MetricsGrid $compact={compact}>
        {metrics.map((metric) => {
          const progress = Math.min((metric.current / metric.goal) * 100, 100);

          return (
            <MetricCard
              key={metric.label}
              $interactive={metric.interactive}
              onClick={metric.onClick}
              title={`${metric.label}: ${metric.value}`}
            >
              <MetricIcon $color={metric.color}>
                {metric.icon}
              </MetricIcon>
              <MetricLabel>{metric.label}</MetricLabel>
              <MetricValue $color={metric.color}>
                {metric.value}
              </MetricValue>
              <ProgressBar $color={metric.color}>
                <ProgressFill $progress={progress} $color={metric.color} />
              </ProgressBar>
              {metric.interactive && metric.action && (
                <MetricAction onClick={(e) => {
                  e.stopPropagation();
                  metric.onClick?.();
                }}>
                  {metric.action}
                </MetricAction>
              )}
            </MetricCard>
          );
        })}
      </MetricsGrid>
    </WellnessContainer>
  );
};

export type { WellnessMetricsProps };