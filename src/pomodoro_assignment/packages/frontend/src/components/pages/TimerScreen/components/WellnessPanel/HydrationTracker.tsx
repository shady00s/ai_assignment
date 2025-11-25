import React, { useState } from 'react';
import styled from 'styled-components';

interface HydrationTrackerProps {
  currentGlasses: number;
  dailyGoal: number;
  glassSize: number; // in ml
  onIncrement?: (glasses: number) => void;
  onDecrement?: (glasses: number) => void;
  onGoalUpdate?: (newGoal: number) => void;
  isLoading?: boolean;
  compact?: boolean;
  className?: string;
}

const HydrationContainer = styled.div<{ $compact: boolean }>`
  background: rgba(255, 255, 255, 0.8);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ $compact, theme }) => $compact ? theme.spacing.mobile.sm : theme.spacing.mobile.md};
  border: 1px solid rgba(74, 144, 226, 0.2);
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ $compact, theme }) => $compact ? theme.spacing.tablet.sm : theme.spacing.tablet.md};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ $compact }) => $compact ? '12px' : '20px'};
    border-radius: 16px;
  }
`;

const HydrationHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
`;

const HydrationTitle = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #4A90E2;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.xs};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 4px;
    font-size: 14px;
  }
`;

const HydrationIcon = styled.span<{ $filled: boolean }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.md};
  opacity: ${({ $filled }) => $filled ? 1 : 0.3};
  transition: all 0.3s ease;
  animation: ${({ $filled }) => $filled ? 'waterDrop 2s ease-in-out infinite' : 'none'};

  @keyframes waterDrop {
    0%, 100% {
      transform: translateY(0) scale(1);
    }
    50% {
      transform: translateY(-2px) scale(1.1);
    }
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 20px;
  }
`;

const HydrationGoal = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: #8B7D7B;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.xs};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 11px;
  }
`;

const HydrationProgress = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: rgba(74, 144, 226, 0.1);
  border-radius: 4px;
  overflow: hidden;
  position: relative;

  ${({ theme }) => theme.mediaQueries.tablet} {
    height: 10px;
    border-radius: 5px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    height: 12px;
    border-radius: 6px;
  }
`;

const ProgressFill = styled.div<{ $progress: number; $animated?: boolean }>`
  height: 100%;
  background: linear-gradient(90deg, #4A90E2 0%, #5BA0F2 100%);
  border-radius: inherit;
  width: ${({ $progress }) => $progress}%;
  transition: width 0.5s ease-in-out;
  position: relative;

  ${({ $animated }) =>
    $animated &&
    `
    &::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
      animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }
  `}
`;

const HydrationStats = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 14px;
    margin-bottom: 12px;
  }
`;

const CurrentAmount = styled.div`
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: #4A90E2;
`;

const TotalAmount = styled.div`
  color: #8B7D7B;
`;

const HydrationControls = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  justify-content: center;

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 8px;
  }
`;

const HydrationButton = styled.button<{ $variant: 'increment' | 'decrement'; $disabled?: boolean }>`
  background: ${({ $variant }) => $variant === 'increment' ? '#4A90E2' : 'transparent'};
  color: ${({ $variant }) => $variant === 'increment' ? 'white' : '#4A90E2'};
  border: 2px solid #4A90E2;
  border-radius: 50%;
  width: 36px;
  height: 36px;
  font-size: 18px;
  font-weight: bold;
  cursor: ${({ $disabled }) => $disabled ? 'not-allowed' : 'pointer'};
  opacity: ${({ $disabled }) => $disabled ? 0.5 : 1};
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover:not(:disabled) {
    transform: ${({ $variant }) => $variant === 'increment' ? 'scale(1.1)' : 'scale(0.95)'};
    box-shadow: ${({ $variant }) => $variant === 'increment' ? '0 4px 12px rgba(74, 144, 226, 0.3)' : 'none'};
  }

  &:active:not(:disabled) {
    transform: scale(0.95);
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    width: 40px;
    height: 40px;
    font-size: 20px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    width: 44px;
    height: 44px;
    font-size: 22px;
  }
`;

const GlassesGrid = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: center;
  margin-top: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: 8px;
    margin-top: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 10px;
    margin-top: 12px;
  }
`;

const GlassIcon = styled.div<{ $filled: boolean; $interactive?: boolean }>`
  font-size: 20px;
  cursor: ${({ $interactive }) => $interactive ? 'pointer' : 'default'};
  opacity: ${({ $filled }) => $filled ? 1 : 0.3};
  transition: all 0.2s ease;
  filter: ${({ $filled }) => $filled ? 'hue-rotate(200deg)' : 'grayscale(1)'};

  &:hover {
    transform: ${({ $interactive }) => $interactive ? 'scale(1.2)' : 'scale(1)'};
    opacity: ${({ $filled, $interactive }) => $filled || !$interactive ? 1 : 0.5};
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 24px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 28px;
  }
`;

const AchievementBadge = styled.div<{ $visible: boolean }>`
  text-align: center;
  margin-top: ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #4A90E2;
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  opacity: ${({ $visible }) => $visible ? 1 : 0};
  transform: ${({ $visible }) => $visible ? 'translateY(0)' : 'translateY(-10px)'};
  transition: all 0.3s ease;

  ${({ theme }) => theme.mediaQueries.tablet} {
    margin-top: ${({ theme }) => theme.spacing.tablet.sm};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    margin-top: 12px;
    font-size: 14px;
  }
`;

export const HydrationTracker: React.FC<HydrationTrackerProps> = ({
  currentGlasses,
  dailyGoal,
  glassSize,
  onIncrement,
  onDecrement,
  onGoalUpdate,
  isLoading = false,
  compact = false,
  className,
}) => {
  const [isAnimating, setIsAnimating] = useState(false);

  const progress = Math.min((currentGlasses / dailyGoal) * 100, 100);
  const totalMl = currentGlasses * glassSize;
  const goalMl = dailyGoal * glassSize;
  const isGoalMet = currentGlasses >= dailyGoal;

  const handleIncrement = () => {
    if (currentGlasses < dailyGoal * 2 && !isLoading) {
      setIsAnimating(true);
      onIncrement?.(currentGlasses + 1);
      setTimeout(() => setIsAnimating(false), 500);
    }
  };

  const handleDecrement = () => {
    if (currentGlasses > 0 && !isLoading) {
      onDecrement?.(currentGlasses - 1);
    }
  };

  const handleGlassClick = (glassNumber: number) => {
    if (!isLoading) {
      if (glassNumber <= currentGlasses) {
        // Clicking a filled glass - decrement to this level
        onDecrement?.(glassNumber - 1);
      } else {
        // Clicking an empty glass - increment to this level
        onIncrement?.(Math.min(glassNumber, dailyGoal * 2));
      }
    }
  };

  return (
    <HydrationContainer $compact={compact} className={className}>
      <HydrationHeader>
        <HydrationTitle>
          <HydrationIcon $filled={currentGlasses > 0}>💧</HydrationIcon>
          Hydration
        </HydrationTitle>
        <HydrationGoal>
          Goal: {dailyGoal} glasses ({goalMl}ml)
        </HydrationGoal>
      </HydrationHeader>

      <HydrationProgress>
        <ProgressBar>
          <ProgressFill $progress={progress} $animated={isAnimating} />
        </ProgressBar>
      </HydrationProgress>

      <HydrationStats>
        <CurrentAmount>
          {totalMl}ml consumed
        </CurrentAmount>
        <TotalAmount>
          {goalMl}ml daily goal
        </TotalAmount>
      </HydrationStats>

      <HydrationControls>
        <HydrationButton
          $variant="decrement"
          onClick={handleDecrement}
          $disabled={currentGlasses === 0 || isLoading}
          aria-label="Decrease hydration"
        >
          −
        </HydrationButton>
        <HydrationButton
          $variant="increment"
          onClick={handleIncrement}
          $disabled={currentGlasses >= dailyGoal * 2 || isLoading}
          aria-label="Increase hydration"
        >
          +
        </HydrationButton>
      </HydrationControls>

      <GlassesGrid>
        {Array.from({ length: Math.max(dailyGoal, currentGlasses + 2) }, (_, i) => i + 1).map((glassNum) => (
          <GlassIcon
            key={glassNum}
            $filled={glassNum <= currentGlasses}
            $interactive={!isLoading}
            onClick={() => handleGlassClick(glassNum)}
            title={`Glass ${glassNum}: ${glassNum <= currentGlasses ? 'Filled' : 'Empty'}`}
          >
            {glassNum <= currentGlasses ? '🥤' : '🥃'}
          </GlassIcon>
        ))}
      </GlassesGrid>

      <AchievementBadge $visible={isGoalMet}>
        🎉 Daily Goal Achieved! Great job staying hydrated!
      </AchievementBadge>
    </HydrationContainer>
  );
};

export type { HydrationTrackerProps };