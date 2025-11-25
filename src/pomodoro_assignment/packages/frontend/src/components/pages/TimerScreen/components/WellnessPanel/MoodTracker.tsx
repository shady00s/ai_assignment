import React from 'react';
import styled from 'styled-components';

interface MoodTrackerProps {
  moodRating: number;
  stressLevel: number;
  energyLevel: number;
  onMoodUpdate?: (mood: number) => void;
  onStressUpdate?: (stress: number) => void;
  onEnergyUpdate?: (energy: number) => void;
  lastCheckIn?: string;
  isLoading?: boolean;
  compact?: boolean;
  className?: string;
}

const MoodContainer = styled.div<{ $compact: boolean }>`
  background: rgba(255, 255, 255, 0.8);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ $compact, theme }) => $compact ? theme.spacing.mobile.sm : theme.spacing.mobile.md};
  border: 1px solid rgba(243, 156, 18, 0.2);
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ $compact, theme }) => $compact ? theme.spacing.tablet.sm : theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ $compact }) => $compact ? '12px' : '20px'};
  }
`;

const MoodTitle = styled.h4`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #F39C12;
  margin: 0 0 ${({ theme }) => theme.spacing.mobile.sm} 0;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
`;

const MoodGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing.mobile.sm};
`;

const MoodItem = styled.div`
  text-align: center;
`;

const MoodLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: #8B7D7B;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const MoodStars = styled.div`
  display: flex;
  justify-content: center;
  gap: 2px;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xs};
`;

const Star = styled.span<{ $active: boolean; $interactive?: boolean }>`
  font-size: 16px;
  cursor: ${({ $interactive }) => $interactive ? 'pointer' : 'default'};
  opacity: ${({ $active }) => $active ? 1 : 0.3};
  transition: all 0.2s ease;

  ${({ $interactive }) =>
    $interactive &&
    `
    &:hover {
      transform: scale(1.2);
      opacity: 1;
    }
  `}
`;

const MoodValue = styled.div<{ $level: number }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ $level }) => {
    if ($level >= 4) return '#7FA870';
    if ($level >= 3) return '#F39C12';
    return '#C85A5A';
  }};
`;

export const MoodTracker: React.FC<MoodTrackerProps> = ({
  moodRating,
  stressLevel,
  energyLevel,
  onMoodUpdate,
  onStressUpdate,
  onEnergyUpdate,
  lastCheckIn,
  isLoading = false,
  compact = false,
  className,
}) => {
  const renderStars = (rating: number, maxRating: number, onUpdate?: (value: number) => void) => {
    return Array.from({ length: maxRating }, (_, i) => i + 1).map((star) => (
      <Star
        key={star}
        $active={star <= rating}
        $interactive={!!onUpdate && !isLoading}
        onClick={() => onUpdate?.(star)}
      >
        ⭐
      </Star>
    ));
  };

  const getLevelText = (level: number, type: 'mood' | 'stress' | 'energy') => {
    if (type === 'mood') {
      if (level >= 5) return 'Excellent';
      if (level >= 4) return 'Good';
      if (level >= 3) return 'Okay';
      if (level >= 2) return 'Low';
      return 'Very Low';
    }
    if (type === 'stress') {
      if (level <= 1) return 'Very Calm';
      if (level <= 2) return 'Calm';
      if (level <= 3) return 'Moderate';
      if (level <= 4) return 'Stressed';
      return 'Very Stressed';
    }
    // energy
    if (level >= 5) return 'Very High';
    if (level >= 4) return 'High';
    if (level >= 3) return 'Medium';
    if (level >= 2) return 'Low';
    return 'Very Low';
  };

  return (
    <MoodContainer $compact={compact} className={className}>
      <MoodTitle>😊 Mood & Energy</MoodTitle>

      <MoodGrid>
        <MoodItem>
          <MoodLabel>Mood</MoodLabel>
          <MoodStars>
            {renderStars(moodRating, 5, onMoodUpdate)}
          </MoodStars>
          <MoodValue $level={moodRating}>
            {getLevelText(moodRating, 'mood')}
          </MoodValue>
        </MoodItem>

        <MoodItem>
          <MoodLabel>Stress</MoodLabel>
          <MoodStars>
            {renderStars(6 - stressLevel, 5, (value) => onStressUpdate?.(6 - value))}
          </MoodStars>
          <MoodValue $level={6 - stressLevel}>
            {getLevelText(stressLevel, 'stress')}
          </MoodValue>
        </MoodItem>

        <MoodItem>
          <MoodLabel>Energy</MoodLabel>
          <MoodStars>
            {renderStars(energyLevel, 5, onEnergyUpdate)}
          </MoodStars>
          <MoodValue $level={energyLevel}>
            {getLevelText(energyLevel, 'energy')}
          </MoodValue>
        </MoodItem>
      </MoodGrid>

      {lastCheckIn && (
        <div style={{
          fontSize: '11px',
          color: '#A8968E',
          textAlign: 'center',
          marginTop: '8px',
        }}>
          Last check-in: {new Date(lastCheckIn).toLocaleTimeString()}
        </div>
      )}
    </MoodContainer>
  );
};

export type { MoodTrackerProps };