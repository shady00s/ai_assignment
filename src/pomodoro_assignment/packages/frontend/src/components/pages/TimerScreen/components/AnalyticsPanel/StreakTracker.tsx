import React from 'react';
import styled from 'styled-components';

interface StreakTrackerProps {
  currentStreak: number;
  longestStreak: number;
  streakHistory: boolean[]; // Last 30 days, true = completed
  todayCompleted: boolean;
  className?: string;
}

const StreakContainer = styled.div`
  background: rgba(255, 255, 255, 0.8);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.mobile.md};
  border: 1px solid rgba(231, 76, 60, 0.1);
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 20px;
    border-radius: 16px;
  }
`;

const StreakHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
`;

const StreakTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #E74C3C;
  margin: 0;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
`;

const StreakFlame = styled.span<{ $active: boolean }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  animation: ${({ $active }) => $active ? 'flicker 2s ease-in-out infinite' : 'none'};
  filter: hue-rotate(0deg);

  @keyframes flicker {
    0%, 100% {
      transform: scale(1) rotate(0deg);
      filter: hue-rotate(0deg) brightness(1);
    }
    25% {
      transform: scale(1.1) rotate(-5deg);
      filter: hue-rotate(10deg) brightness(1.2);
    }
    50% {
      transform: scale(1.05) rotate(5deg);
      filter: hue-rotate(-10deg) brightness(1.1);
    }
    75% {
      transform: scale(1.1) rotate(-3deg);
      filter: hue-rotate(5deg) brightness(1.3);
    }
  }
`;

const StreakStats = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.md};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.sm};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 16px;
    margin-bottom: 20px;
  }
`;

const StreakStat = styled.div<{ $primary?: boolean }>`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  background: ${({ $primary }) => $primary ? 'rgba(231, 76, 60, 0.1)' : 'rgba(255, 255, 255, 0.6)'};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border: 1px solid ${({ $primary }) => $primary ? 'rgba(231, 76, 60, 0.2)' : 'rgba(200, 200, 200, 0.2)'};
`;

const StreakValue = styled.div<{ $color: string }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ $color }) => $color};
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 24px;
  }
`;

const StreakLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: #8B7D7B;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  margin-top: 2px;
`;

const CalendarGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 4px;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 6px;
  }
`;

const DayHeader = styled.div`
  text-align: center;
  font-size: 10px;
  color: #A8968E;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  padding: 2px 0;

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 11px;
  }
`;

const DayCell = styled.div<{ $completed: boolean; $today?: boolean }>`
  aspect-ratio: 1;
  border-radius: 4px;
  background: ${({ $completed, $today }) => {
    if ($today) return $completed ? '#E74C3C' : 'rgba(231, 76, 60, 0.2)';
    return $completed ? 'rgba(231, 76, 60, 0.8)' : 'rgba(200, 200, 200, 0.3)';
  }};
  border: 1px solid ${({ $today }) => $today ? '#E74C3C' : 'transparent'};
  position: relative;
  cursor: default;
  transition: all 0.2s ease;

  &:hover {
    transform: scale(1.1);
    box-shadow: ${({ $completed }) => $completed ? '0 4px 8px rgba(231, 76, 60, 0.3)' : 'none'};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    border-radius: 6px;
  }
`;

const DayNumber = styled.div<{ $visible: boolean }>`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 8px;
  color: white;
  font-weight: bold;
  opacity: ${({ $visible }) => $visible ? 1 : 0};

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 9px;
  }
`;

const StreakMessage = styled.div<{ $active: boolean }>`
  text-align: center;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: ${({ $active }) => $active ? '#E74C3C' : '#A8968E'};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  font-style: ${({ $active }) => $active ? 'normal' : 'italic'};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 14px;
  }
`;

const MilestoneBadges = styled.div`
  display: flex;
  gap: 6px;
  justify-content: center;
  margin-top: ${({ theme }) => theme.spacing.mobile.sm};
  flex-wrap: wrap;

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 8px;
    margin-top: 12px;
  }
`;

const MilestoneBadge = styled.div`
  background: linear-gradient(135deg, #E74C3C 0%, #EC7063 100%);
  color: white;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 10px;
  font-weight: bold;
  display: flex;
  align-items: center;
  gap: 2px;
`;

export const StreakTracker: React.FC<StreakTrackerProps> = ({
  currentStreak,
  longestStreak,
  streakHistory,
  todayCompleted,
  className,
}) => {
  // Get last 30 days
  const getDayData = () => {
    const days = [];
    const today = new Date();

    for (let i = 29; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      days.push({
        date,
        dayOfMonth: date.getDate(),
        dayName: date.toLocaleDateString('en', { weekday: 'short' }).charAt(0),
        completed: streakHistory[29 - i] || false,
        isToday: i === 0,
      });
    }

    return days;
  };

  const days = getDayData();
  const dayHeaders = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

  const getStreakMessage = () => {
    if (currentStreak === 0) {
      return "Start your streak today! Complete a focus session to begin.";
    } else if (currentStreak < 3) {
      return `Great start! ${3 - currentStreak} more days to reach your first milestone.`;
    } else if (currentStreak < 7) {
      return `Keep it going! ${7 - currentStreak} days until your weekly streak.`;
    } else if (currentStreak < 14) {
      return `Impressive consistency! You're on a ${currentStreak}-day roll!`;
    } else if (currentStreak < 30) {
      return `Outstanding dedication! ${30 - currentStreak} days from your monthly goal!`;
    } else {
      return `Legendary streak! ${currentStreak} days of continuous focus! 🔥`;
    }
  };

  const getMilestones = () => {
    const milestones = [];
    if (currentStreak >= 3) milestones.push({ number: 3, icon: '🌱', text: 'Sprout' });
    if (currentStreak >= 7) milestones.push({ number: 7, icon: '🌿', text: 'Week' });
    if (currentStreak >= 14) milestones.push({ number: 14, icon: '🎋', text: 'Fortnight' });
    if (currentStreak >= 21) milestones.push({ number: 21, icon: '🌴', text: 'Triumph' });
    if (currentStreak >= 30) milestones.push({ number: 30, icon: '🏆', text: 'Month' });
    if (currentStreak === longestStreak && longestStreak >= 10) {
      milestones.push({ number: longestStreak, icon: '👑', text: 'Personal Best' });
    }
    return milestones;
  };

  return (
    <StreakContainer className={className}>
      <StreakHeader>
        <StreakTitle>
          <StreakFlame $active={currentStreak > 0}>🔥</StreakFlame>
          Streak Tracker
        </StreakTitle>
      </StreakHeader>

      <StreakStats>
        <StreakStat $primary>
          <StreakValue $color="#E74C3C">{currentStreak}</StreakValue>
          <StreakLabel>Current Streak</StreakLabel>
        </StreakStat>
        <StreakStat>
          <StreakValue $color="#F39C12">{longestStreak}</StreakValue>
          <StreakLabel>Longest Streak</StreakLabel>
        </StreakStat>
      </StreakStats>

      <CalendarGrid>
        {dayHeaders.map((day, index) => (
          <DayHeader key={index}>{day}</DayHeader>
        ))}
        {days.map((day, index) => (
          <DayCell
            key={index}
            $completed={day.completed}
            $today={day.isToday}
            title={`${day.date.toLocaleDateString()}: ${day.completed ? 'Completed' : 'Missed'}`}
          >
            <DayNumber $visible={day.dayOfMonth <= 7 || day.isToday}>
              {day.dayOfMonth <= 7 ? day.dayOfMonth : ''}
            </DayNumber>
          </DayCell>
        ))}
      </CalendarGrid>

      <StreakMessage $active={currentStreak > 0}>
        {getStreakMessage()}
      </StreakMessage>

      {currentStreak > 0 && (
        <MilestoneBadges>
          {getMilestones().map((milestone, index) => (
            <MilestoneBadge key={index}>
              {milestone.icon} {milestone.text}
            </MilestoneBadge>
          ))}
        </MilestoneBadges>
      )}
    </StreakContainer>
  );
};

export type { StreakTrackerProps };