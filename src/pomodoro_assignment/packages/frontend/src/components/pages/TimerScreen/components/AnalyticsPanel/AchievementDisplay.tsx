import React, { useState } from 'react';
import styled from 'styled-components';

interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: 'FOCUS' | 'CONSISTENCY' | 'WELLNESS' | 'COLLABORATION' | 'MILESTONES';
  unlockedAt: string;
  progress: number; // 0-100
  isNew?: boolean;
}

interface AchievementDisplayProps {
  achievements: Achievement[];
  onAchievementClick?: (achievement: Achievement) => void;
  compact?: boolean;
  maxDisplay?: number;
  className?: string;
}

const AchievementContainer = styled.div<{ $compact: boolean }>`
  background: rgba(255, 255, 255, 0.8);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ $compact, theme }) => $compact ? theme.spacing.mobile.sm : theme.spacing.mobile.md};
  border: 1px solid rgba(233, 196, 106, 0.2);
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ $compact, theme }) => $compact ? theme.spacing.tablet.sm : theme.spacing.tablet.md};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ $compact }) => $compact ? '16px' : '24px'};
    border-radius: 20px;
  }
`;

const AchievementHeader = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.md};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #E9C46A;
  margin: 0 0 ${({ theme }) => theme.spacing.mobile.sm} 0;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.md};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.sm};
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 18px;
    margin-bottom: 16px;
    gap: 8px;
  }
`;

const AchievementGrid = styled.div<{ $compact: boolean }>`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(${({ $compact }) => $compact ? '100px' : '120px'}, 1fr));
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    grid-template-columns: repeat(auto-fill, minmax(${({ $compact }) => $compact ? '110px' : '140px'}, 1fr));
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    grid-template-columns: repeat(auto-fill, minmax(${({ $compact }) => $compact ? '120px' : '160px'}, 1fr));
    gap: 12px;
  }
`;

const AchievementCard = styled.div<{ $unlocked: boolean; $new?: boolean; $interactive?: boolean }>`
  background: ${({ $unlocked }) => $unlocked ? 'rgba(233, 196, 106, 0.1)' : 'rgba(200, 200, 200, 0.1)'};
  border: ${({ $unlocked }) => $unlocked ? '2px solid rgba(233, 196, 106, 0.3)' : '2px solid rgba(200, 200, 200, 0.3)'};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  text-align: center;
  cursor: ${({ $interactive }) => $interactive ? 'pointer' : 'default'};
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;

  ${({ $interactive }) =>
    $interactive &&
    `
    &:hover {
      transform: translateY(-4px) scale(1.05);
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
  `}

  ${({ $new }) =>
    $new &&
    `
    &::before {
      content: 'NEW!';
      position: absolute;
      top: 4px;
      right: 4px;
      background: #E9C46A;
      color: white;
      font-size: 8px;
      font-weight: bold;
      padding: 2px 4px;
      border-radius: 4px;
      z-index: 2;
      animation: badgePulse 2s ease-in-out infinite;
    }

    @keyframes badgePulse {
      0%, 100% {
        transform: scale(1);
        opacity: 1;
      }
      50% {
        transform: scale(1.1);
        opacity: 0.8;
      }
    }
  `}

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.sm};
    border-radius: ${({ theme }) => theme.borderRadius.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 16px;
    border-radius: 16px;
  }
`;

const AchievementIcon = styled.div<{ $unlocked: boolean; $new?: boolean }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile['2xl']};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
  opacity: ${({ $unlocked }) => $unlocked ? 1 : 0.3};
  filter: ${({ $unlocked, $new }) => $unlocked ? ($new ? 'hue-rotate(20deg) brightness(1.2)' : 'none') : 'grayscale(1)'};
  animation: ${({ $new }) => $new ? 'achievementGlow 3s ease-in-out infinite' : 'none'};
  transition: all 0.3s ease;

  @keyframes achievementGlow {
    0%, 100% {
      filter: hue-rotate(20deg) brightness(1.2) drop-shadow(0 0 8px rgba(233, 196, 106, 0.5));
    }
    50% {
      filter: hue-rotate(30deg) brightness(1.4) drop-shadow(0 0 12px rgba(233, 196, 106, 0.7));
    }
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 36px;
    margin-bottom: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 40px;
    margin-bottom: 8px;
  }
`;

const AchievementName = styled.div<{ $unlocked: boolean }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ $unlocked }) => $unlocked ? '#2C3E50' : '#A8968E'};
  margin-bottom: 2px;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  line-height: 1.2;

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 13px;
  }
`;

const AchievementDescription = styled.div<{ $unlocked: boolean }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: ${({ $unlocked }) => $unlocked ? '#8B7D7B' : '#C8BDB8'};
  line-height: 1.3;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.xs};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 11px;
  }
`;

const AchievementProgress = styled.div<{ $progress: number; $show: boolean }>`
  width: 100%;
  height: 4px;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 2px;
  overflow: hidden;
  margin-top: ${({ theme }) => theme.spacing.mobile.xs};
  opacity: ${({ $show }) => $show ? 1 : 0};

  ${({ theme }) => theme.mediaQueries.desktop} {
    height: 6px;
    border-radius: 3px;
    margin-top: 6px;
  }
`;

const ProgressFill = styled.div<{ $progress: number }>`
  height: 100%;
  background: linear-gradient(90deg, #E9C46A 0%, #F4A261 100%);
  border-radius: inherit;
  width: ${({ $progress }) => $progress}%;
  transition: width 0.5s ease-in-out;
`;

const ViewAllButton = styled.button`
  width: 100%;
  background: transparent;
  color: #E9C46A;
  border: 1px solid rgba(233, 196, 106, 0.3);
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all 0.2s ease;
  margin-top: ${({ theme }) => theme.spacing.mobile.sm};

  &:hover {
    background: rgba(233, 196, 106, 0.1);
    transform: translateY(-1px);
  }
`;

export const AchievementDisplay: React.FC<AchievementDisplayProps> = ({
  achievements,
  onAchievementClick,
  compact = false,
  maxDisplay = 6,
  className,
}) => {
  const [showAll, setShowAll] = useState(false);
  const displayAchievements = showAll ? achievements : achievements.slice(0, maxDisplay);

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'FOCUS': return '#E67E50';
      case 'CONSISTENCY': return '#7FA870';
      case 'WELLNESS': return '#4A90E2';
      case 'COLLABORATION': return '#9B59B6';
      case 'MILESTONES': return '#E9C46A';
      default: return '#A8968E';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    return `${Math.floor(diffDays / 30)} months ago`;
  };

  return (
    <AchievementContainer $compact={compact} className={className}>
      <AchievementHeader>
        🏆 Recent Achievements
      </AchievementHeader>

      <AchievementGrid $compact={compact}>
        {displayAchievements.map((achievement) => (
          <AchievementCard
            key={achievement.id}
            $unlocked={achievement.progress === 100}
            $new={achievement.isNew}
            $interactive={!!onAchievementClick}
            onClick={() => onAchievementClick?.(achievement)}
            title={`${achievement.name}: ${achievement.description}`}
          >
            <AchievementIcon
              $unlocked={achievement.progress === 100}
              $new={achievement.isNew}
            >
              {achievement.icon}
            </AchievementIcon>
            <AchievementName $unlocked={achievement.progress === 100}>
              {achievement.name}
            </AchievementName>
            <AchievementDescription $unlocked={achievement.progress === 100}>
              {achievement.description}
            </AchievementDescription>
            {achievement.progress === 100 && (
              <div style={{
                fontSize: '10px',
                color: '#A8968E',
                marginTop: '4px',
              }}>
                {formatDate(achievement.unlockedAt)}
              </div>
            )}
            <AchievementProgress
              $progress={achievement.progress}
              $show={achievement.progress < 100 && achievement.progress > 0}
            >
              <ProgressFill $progress={achievement.progress} />
            </AchievementProgress>
          </AchievementCard>
        ))}
      </AchievementGrid>

      {achievements.length > maxDisplay && (
        <ViewAllButton onClick={() => setShowAll(!showAll)}>
          {showAll ? 'Show Less' : `View All (${achievements.length})`}
        </ViewAllButton>
      )}
    </AchievementContainer>
  );
};

export type { AchievementDisplayProps, Achievement };