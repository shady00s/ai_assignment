import React from 'react';
import styled from 'styled-components';

interface EnhancedControlsProps {
  isRunning: boolean;
  isPaused: boolean;
  canStart: boolean;
  canPause: boolean;
  canSkip: boolean;
  canComplete: boolean;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onSkip: () => void;
  onComplete: () => void;
  sessionType: 'POMODORO' | 'SHORT_BREAK' | 'LONG_BREAK';
  className?: string;
}

const ControlsContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.md};
  flex-wrap: wrap;
  padding: ${({ theme }) => theme.spacing.mobile.md};
  background: rgba(255, 255, 255, 0.6);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 1px solid rgba(127, 168, 112, 0.1);
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.lg};
    padding: ${({ theme }) => theme.spacing.tablet.md};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 20px;
    padding: 24px;
    border-radius: 20px;
  }
`;

const ControlButton = styled.button<{ $variant: 'primary' | 'secondary' | 'warning' | 'danger'; $disabled?: boolean }>`
  padding: ${({ theme }) => theme.spacing.mobile.lg} ${({ theme }) => theme.spacing.mobile.xl};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  min-width: 120px;
  height: 60px;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  cursor: ${({ $disabled }) => $disabled ? 'not-allowed' : 'pointer'};
  opacity: ${({ $disabled }) => $disabled ? 0.6 : 1};
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s ease;
  }

  &:hover:not(:disabled) {
    transform: translateY(-2px);
    &::before {
      left: 100%;
    }
  }

  &:active:not(:disabled) {
    transform: translateY(0);
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.lg} ${({ theme }) => theme.spacing.tablet.xl};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.lg};
    min-width: 140px;
    height: 64px;
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 18px 36px;
    font-size: 18px;
    min-width: 160px;
    height: 68px;
    gap: 8px;
  }

  ${({ $variant }) => {
    switch ($variant) {
      case 'primary':
        return `
          background: linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%);
          color: white;
          border: none;
          box-shadow: 0 6px 20px rgba(127, 168, 112, 0.3);

          &:hover:not(:disabled) {
            box-shadow: 0 8px 25px rgba(127, 168, 112, 0.4);
          }
        `;
      case 'warning':
        return `
          background: linear-gradient(135deg, #F4A261 0%, #F5B789 100%);
          color: white;
          border: none;
          box-shadow: 0 6px 20px rgba(244, 162, 97, 0.3);

          &:hover:not(:disabled) {
            box-shadow: 0 8px 25px rgba(244, 162, 97, 0.4);
          }
        `;
      case 'danger':
        return `
          background: linear-gradient(135deg, #C85A5A 0%, #D57A7A 100%);
          color: white;
          border: none;
          box-shadow: 0 6px 20px rgba(200, 90, 90, 0.3);

          &:hover:not(:disabled) {
            box-shadow: 0 8px 25px rgba(200, 90, 90, 0.4);
          }
        `;
      case 'secondary':
        return `
          background: transparent;
          color: #8B7D7B;
          border: 2px solid #D4C4B0;

          &:hover:not(:disabled) {
            background: rgba(212, 196, 176, 0.1);
            border-color: #8B7D7B;
            color: #2C3E50;
          }
        `;
      default:
        return '';
    }
  }}
`;

const ButtonIcon = styled.span<{ $pulse?: boolean }>`
  font-size: 1.2em;
  animation: ${({ $pulse }) => $pulse ? 'pulse 2s ease-in-out infinite' : 'none'};

  @keyframes pulse {
    0%, 100% {
      transform: scale(1);
    }
    50% {
      transform: scale(1.1);
    }
  }
`;

const ButtonText = styled.span`
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
`;

const QuickActions = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  margin-top: ${({ theme }) => theme.spacing.mobile.sm};
  justify-content: center;
  flex-wrap: wrap;

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.sm};
    margin-top: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 8px;
    margin-top: 12px;
  }
`;

const QuickActionButton = styled.button`
  background: rgba(127, 168, 112, 0.1);
  border: 1px solid rgba(127, 168, 112, 0.2);
  border-radius: 20px;
  padding: 6px 12px;
  font-size: 12px;
  color: #7FA870;
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};

  &:hover {
    background: rgba(127, 168, 112, 0.2);
    transform: translateY(-1px);
  }
`;

export const EnhancedControls: React.FC<EnhancedControlsProps> = ({
  isRunning,
  isPaused,
  canStart,
  canPause,
  canSkip,
  canComplete,
  onStart,
  onPause,
  onResume,
  onSkip,
  onComplete,
  sessionType,
  className,
}) => {
  const getSessionButtonText = () => {
    if (isPaused) return 'Resume';
    if (isRunning) return 'Pause';
    return 'Start';
  };

  const getSessionButtonIcon = () => {
    if (isPaused) return '▶️';
    if (isRunning) return '⏸️';
    return '▶️';
  };

  const getSessionButtonVariant = () => {
    if (isPaused || !isRunning) return 'primary' as const;
    return 'warning' as const;
  };

  const getSessionButtonAction = () => {
    if (isPaused) return onResume;
    if (isRunning) return onPause;
    return onStart;
  };

  return (
    <ControlsContainer className={className}>
      <ControlButton
        $variant={getSessionButtonVariant()}
        onClick={getSessionButtonAction()}
        $disabled={!canStart && !isPaused}
      >
        <ButtonIcon $pulse={!isRunning && canStart}>
          {getSessionButtonIcon()}
        </ButtonIcon>
        <ButtonText>{getSessionButtonText()}</ButtonText>
      </ControlButton>

      <ControlButton
        $variant="secondary"
        onClick={onSkip}
        $disabled={!canSkip}
      >
        <ButtonIcon>⏭️</ButtonIcon>
        <ButtonText>Skip</ButtonText>
      </ControlButton>

      <ControlButton
        $variant="danger"
        onClick={onComplete}
        $disabled={!canComplete}
      >
        <ButtonIcon>✅</ButtonIcon>
        <ButtonText>Complete</ButtonText>
      </ControlButton>

      <QuickActions>
        <QuickActionButton>
          🔇 Focus Mode
        </QuickActionButton>
        <QuickActionButton>
          🎵 Ambient Sounds
        </QuickActionButton>
        <QuickActionButton>
          📊 Session Stats
        </QuickActionButton>
      </QuickActions>
    </ControlsContainer>
  );
};

export type { EnhancedControlsProps };