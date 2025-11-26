import React from 'react';
import styled from 'styled-components';
import { Card } from '@/components/atoms';

import { UserAchievement } from '../../../../../types';
import { getLevelColor, getXpProgress } from '../../utils/dataFormatters';

interface AchievementGalleryProps {
  level: number;
  xp: number;
  achievements: UserAchievement[];
}

const AchievementContainer = styled(Card)`
  padding: ${({ theme }) => theme.spacing.lg};
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

const LevelDisplay = styled.div<{ levelColor: string }>`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.sm};
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
  background-color: ${({ levelColor }) => `${levelColor}20`};
  border: 1px solid ${({ levelColor }) => `${levelColor}40`};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  color: ${({ levelColor }) => levelColor};

  .level-icon {
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
  }

  .level-text {
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
  }
`;

const AchievementGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: ${({ theme }) => theme.spacing.md};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const AchievementBadge = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: ${({ theme }) => theme.spacing.md};
  background-color: ${({ theme }) => theme.colors.neutral[50]};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 2px solid ${({ theme }) => theme.colors.neutral[200]};
  transition: all 0.2s ease;

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary.main};
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.md};
  }
`;

const AchievementIcon = styled.div<{ unlocked: boolean }>`
  font-size: 40px;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
  opacity: ${({ unlocked }) => unlocked ? 1 : 0.3};
  filter: grayscale(${({ unlocked }) => unlocked ? 0 : 1});
`;

const AchievementName = styled.div`
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  line-height: 1.2;
`;

const AchievementDescription = styled.div`
  color: ${({ theme }) => theme.colors.neutral[400]};
  font-size: 10px;
  margin-top: 2px;
  line-height: 1.2;
`;

const XPProgressSection = styled.div`
  padding-top: ${({ theme }) => theme.spacing.md};
  border-top: 1px solid ${({ theme }) => theme.colors.neutral[200]};
`;

const XPProgressHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.sm};

  .xp-label {
    color: ${({ theme }) => theme.colors.neutral[500]};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  }

  .xp-value {
    color: ${({ theme }) => theme.colors.neutral[400]};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
  }
`;

const ProgressBarContainer = styled.div`
  width: 100%;
  height: 12px;
  background-color: ${({ theme }) => theme.colors.neutral[200]};
  border-radius: 6px;
  overflow: hidden;
`;

const ProgressBar = styled.div<{ percentage: number; color: string }>`
  height: 100%;
  width: ${({ percentage }) => percentage}%;
  background-color: ${({ color }) => color};
  border-radius: 6px;
  transition: width 0.3s ease-in-out;
  position: relative;

  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
      90deg,
      transparent,
      rgba(255, 255, 255, 0.3),
      transparent
    );
    animation: shimmer 2s infinite;
  }

  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
`;

const EmptyState = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.xl} ${({ theme }) => theme.spacing.lg};
  color: ${({ theme }) => theme.colors.neutral[400]};

  .empty-icon {
    font-size: 48px;
    margin-bottom: ${({ theme }) => theme.spacing.md};
    opacity: 0.5;
  }

  .empty-text {
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    line-height: 1.4;
  }
`;

export const AchievementGallery: React.FC<AchievementGalleryProps> = (props) => {
  const { level, xp, achievements = [] } = props;
  const displayAchievements = achievements;
  const levelColor = getLevelColor(level);
  const xpProgress = getXpProgress(xp, level);

  return (
    <AchievementContainer>
      <CardHeader>
        <h3>Recent Achievements</h3>
        <LevelDisplay levelColor={levelColor}>
          <span className="level-icon">⭐</span>
          <span className="level-text">Level {level}</span>
        </LevelDisplay>
      </CardHeader>

      {displayAchievements.length > 0 ? (
        <AchievementGrid>
          {displayAchievements.map((achievement) => (
            <AchievementBadge key={achievement.id}>
              <AchievementIcon unlocked={true}>
                {achievement.achievement.icon}
              </AchievementIcon>
              <AchievementName>{achievement.achievement.name}</AchievementName>
              <AchievementDescription>
                {achievement.achievement.description}
              </AchievementDescription>
            </AchievementBadge>
          ))}
        </AchievementGrid>
      ) : (
        <EmptyState>
          <div className="empty-icon">🏆</div>
          <div className="empty-text">
            Complete focus sessions and wellness activities to unlock achievements!
          </div>
        </EmptyState>
      )}

      <XPProgressSection>
        <XPProgressHeader>
          <span className="xp-label">Level Progress</span>
          <span className="xp-value">{xpProgress.current} / {xpProgress.total} XP</span>
        </XPProgressHeader>
        <ProgressBarContainer>
          <ProgressBar
            percentage={xpProgress.percentage}
            color={levelColor}
          />
        </ProgressBarContainer>
      </XPProgressSection>
    </AchievementContainer>
  );
};