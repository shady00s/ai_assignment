import { CircularTimer } from '@/components/organisms';
import React, { useMemo } from 'react';
import styled from 'styled-components';

interface ZenGardenTimerProps {
  remainingTime: number;
  totalTime: number;
  sessionType: 'POMODORO' | 'SHORT_BREAK' | 'LONG_BREAK';
  isRunning: boolean;
  isPaused: boolean;
  progress: number;
  sessionsCompleted: number;
  className?: string;
}

const ZenGardenContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.lg};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 32px;
  }
`;

const TimerWrapper = styled.div`
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.md};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 24px;
  }
`;

const ZenElementsContainer = styled.div`
  display: flex;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.mobile.md};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.md};
  flex-wrap: wrap;

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.md};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 24px;
    margin-bottom: 24px;
  }
`;

const ZenElement = styled.div<{ $active: boolean; $progress: number }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile['2xl']};
  opacity: ${({ $active, $progress }) => $active ? 0.3 + ($progress * 0.7) : 0.15};
  transition: all 0.5s ease;
  cursor: default;
  animation: ${({ $active }) => $active ? 'zenPulse 3s ease-in-out infinite' : 'none'};
  filter: blur(${({ $active }) => $active ? '0px' : '1px'});

  &:hover {
    opacity: ${({ $active, $progress }) => $active ? 0.5 + ($progress * 0.5) : 0.3};
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 36px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 42px;
  }

  @keyframes zenPulse {
    0%, 100% {
      transform: scale(1);
      filter: brightness(1);
    }
    50% {
      transform: scale(1.1);
      filter: brightness(1.2);
    }
  }
`;

const SessionInfo = styled.div`
  text-align: center;
  background: linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%);
  padding: ${({ theme }) => theme.spacing.mobile.md} ${({ theme }) => theme.spacing.mobile.xl};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  color: white;
  box-shadow: 0 4px 16px rgba(127, 168, 112, 0.25);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md} ${({ theme }) => theme.spacing.tablet.xl};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 16px 32px;
    border-radius: 20px;
  }
`;

const SessionLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  opacity: 0.9;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xs};
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 13px;
  }
`;

const SessionValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet['3xl']};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 28px;
  }
`;

const ZenGardenStatus = styled.div`
  text-align: center;
  margin-top: ${({ theme }) => theme.spacing.mobile.md};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #8B7D7B;
  font-style: italic;

  ${({ theme }) => theme.mediaQueries.tablet} {
    margin-top: ${({ theme }) => theme.spacing.tablet.md};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    margin-top: 16px;
    font-size: 14px;
  }
`;

export const ZenGardenTimer: React.FC<ZenGardenTimerProps> = ({
  remainingTime,
  totalTime,
  sessionType,
  isRunning,
  isPaused,
  progress,
  sessionsCompleted,
  className,
}) => {
  // Calculate individual element progress based on overall progress
  const zenElements = useMemo(() => {
    const elementCount = 4;
    const activeElements = Math.floor(progress * elementCount) + 1;

    return [
      { emoji: '🌿', name: 'Bamboo', active: true, progress: Math.min(progress * 1.2, 1) },
      { emoji: '🪨', name: 'Stone', active: progress > 0.15, progress: Math.max(0, (progress - 0.15) * 1.18) },
      { emoji: '💧', name: 'Water', active: progress > 0.3, progress: Math.max(0, (progress - 0.3) * 1.43) },
      { emoji: '🎋', name: 'Leaves', active: progress > 0.5, progress: Math.max(0, (progress - 0.5) * 2) },
    ].map((element, index) => ({
      ...element,
      active: index < activeElements && isRunning,
    }));
  }, [progress, isRunning]);

  const getSessionStatus = () => {
    if (!isRunning && !isPaused) return 'Ready to begin your journey';
    if (isPaused) return 'Taking a mindful pause';
    if (sessionType === 'POMODORO') return 'Deep focus in progress';
    return 'Rejuvenating break time';
  };

  const getSessionTypeDisplay = () => {
    switch (sessionType) {
      case 'POMODORO':
        return { label: 'Focus Sessions', icon: '🍅' };
      case 'SHORT_BREAK':
        return { label: 'Short Breaks', icon: '☕' };
      case 'LONG_BREAK':
        return { label: 'Long Breaks', icon: '🌿' };
      default:
        return { label: 'Sessions', icon: '⏰' };
    }
  };

  const sessionDisplay = getSessionTypeDisplay();

  return (
    <ZenGardenContainer className={className}>
      <ZenElementsContainer>
        {zenElements.map((element, index) => (
          <ZenElement
            key={element.name}
            $active={element.active}
            $progress={element.progress}
            title={element.name}
            style={{
              transform: `scale(${0.8 + element.progress * 0.4})`,
              filter: `blur(${element.active ? 0 : 1}px) brightness(${0.7 + element.progress * 0.3})`,
            }}
          >
            {element.emoji}
          </ZenElement>
        ))}
      </ZenElementsContainer>

      <TimerWrapper>
        <CircularTimer
          size={280}
          strokeWidth={12}
          showControls={false}
          progress={progress}
          remainingTime={remainingTime}
          sessionType={sessionType}
        />
      </TimerWrapper>

      <SessionInfo>
        <SessionLabel>
          {sessionDisplay.icon} {sessionDisplay.label} Completed Today
        </SessionLabel>
        <SessionValue>{sessionsCompleted}</SessionValue>
      </SessionInfo>

      <ZenGardenStatus>
        {getSessionStatus()}
      </ZenGardenStatus>
    </ZenGardenContainer>
  );
};

export type { ZenGardenTimerProps };