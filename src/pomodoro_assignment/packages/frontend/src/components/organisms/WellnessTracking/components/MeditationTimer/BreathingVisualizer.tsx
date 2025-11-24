import React from 'react';
import styled, { keyframes } from 'styled-components';

interface BreathingVisualizerProps {
  phase: 'inhale' | 'hold' | 'exhale';
  compact?: boolean;
}

const breatheIn = keyframes`
  0% {
    transform: scale(0.6);
    opacity: 0.3;
  }
  50% {
    transform: scale(1);
    opacity: 1;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

const breatheOut = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(0.6);
    opacity: 0.3;
  }
  100% {
    transform: scale(0.6);
    opacity: 0.3;
  }
`;

const hold = keyframes`
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.05);
    opacity: 0.9;
  }
`;

const VisualizerContainer = styled.div<{ compact?: boolean }>`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.md};
  margin: ${({ theme, compact }) => compact ? theme.spacing.md : theme.spacing.lg} 0;
`;

const BreathingCircle = styled.div<{ phase: 'inhale' | 'hold' | 'exhale'; compact?: boolean }>`
  width: ${({ compact }) => compact ? '80px' : '120px'};
  height: ${({ compact }) => compact ? '80px' : '120px'};
  border-radius: 50%;
  background: radial-gradient(
    circle,
    ${({ theme }) => theme.colors.sageGreen}20 0%,
    ${({ theme }) => theme.colors.sageGreen}10 50%,
    transparent 100%
  );
  border: 3px solid ${({ theme }) => theme.colors.sageGreen}50;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  animation: ${({ phase }) => {
    switch (phase) {
      case 'inhale': return `${breatheIn} 4s ease-in-out infinite`;
      case 'exhale': return `${breatheOut} 4s ease-in-out infinite`;
      case 'hold': return `${hold} 4s ease-in-out infinite`;
    }
  }};

  &::before {
    content: '';
    position: absolute;
    width: 80%;
    height: 80%;
    border-radius: 50%;
    background: radial-gradient(
      circle,
      ${({ theme }) => theme.colors.sageGreen}40 0%,
      transparent 70%
    );
    animation: ${({ phase }) => {
      switch (phase) {
        case 'inhale': return `${breatheIn} 4s ease-in-out infinite 0.5s`;
        case 'exhale': return `${breatheOut} 4s ease-in-out infinite 0.5s`;
        case 'hold': return `${hold} 4s ease-in-out infinite 0.5s`;
      }
    }};
  }

  &::after {
    content: '';
    position: absolute;
    width: 60%;
    height: 60%;
    border-radius: 50%;
    background: radial-gradient(
      circle,
      ${({ theme }) => theme.colors.sageGreen}60 0%,
      transparent 60%
    );
    animation: ${({ phase }) => {
      switch (phase) {
        case 'inhale': return `${breatheIn} 4s ease-in-out infinite 1s`;
        case 'exhale': return `${breatheOut} 4s ease-in-out infinite 1s`;
        case 'hold': return `${hold} 4s ease-in-out infinite 1s`;
      }
    }};
  }
`;

const PhaseText = styled.div<{ compact?: boolean }>`
  font-size: ${({ theme, compact }) => compact ? theme.typography.fontSize.lg : theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.sageGreen};
  text-align: center;
  animation: pulse 2s ease-in-out infinite;

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.7;
    }
  }
`;

const InstructionText = styled.div<{ compact?: boolean }>`
  font-size: ${({ theme, compact }) => compact ? theme.typography.fontSize.sm : theme.typography.fontSize.base};
  color: ${({ theme }) => theme.colors.neutral[600]};
  text-align: center;
  margin-top: -8px;
`;

const CountDown = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.sageGreen};
  font-variant-numeric: tabular-nums;
`;

const phaseMessages = {
  inhale: 'Breathe In',
  hold: 'Hold',
  exhale: 'Breathe Out',
};

const phaseInstructions = {
  inhale: 'Slowly inhale through your nose',
  hold: 'Hold your breath gently',
  exhale: 'Slowly exhale through your mouth',
};

export const BreathingVisualizer: React.FC<BreathingVisualizerProps> = ({
  phase,
  compact = false,
}) => {
  return (
    <VisualizerContainer compact={compact}>
      <BreathingCircle phase={phase} compact={compact}>
        <CountDown>4</CountDown>
      </BreathingCircle>

      <PhaseText compact={compact}>
        {phaseMessages[phase]}
      </PhaseText>

      {!compact && (
        <InstructionText>
          {phaseInstructions[phase]}
        </InstructionText>
      )}
    </VisualizerContainer>
  );
};