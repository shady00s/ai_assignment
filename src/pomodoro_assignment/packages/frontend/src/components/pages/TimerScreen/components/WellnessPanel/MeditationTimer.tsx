import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

interface MeditationOption {
  id: string;
  name: string;
  duration: number;
  type: 'GUIDED' | 'BREATHING' | 'MINDFULNESS';
  audioUrl?: string;
  description?: string;
}

interface MeditationTimerProps {
  totalMinutes: number;
  sessionGoal: number;
  onStartSession: (duration: number) => void;
  onCompleteSession: (duration: number, quality: number) => void;
  guidedOptions: MeditationOption[];
  isLoading?: boolean;
  compact?: boolean;
  className?: string;
}

const MeditationContainer = styled.div<{ $compact: boolean }>`
  background: rgba(255, 255, 255, 0.8);
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ $compact, theme }) => $compact ? theme.spacing.mobile.sm : theme.spacing.mobile.md};
  border: 1px solid rgba(155, 89, 182, 0.2);
  backdrop-filter: blur(10px);

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ $compact, theme }) => $compact ? theme.spacing.tablet.sm : theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ $compact }) => $compact ? '12px' : '20px'};
  }
`;

const MeditationTitle = styled.h4`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #9B59B6;
  margin: 0 0 ${({ theme }) => theme.spacing.mobile.sm} 0;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
`;

const MeditationStats = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
`;

const StatLabel = styled.span`
  color: #8B7D7B;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const StatValue = styled.span<{ $color: string }>`
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ $color }) => $color};
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 6px;
  background: rgba(155, 89, 182, 0.1);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
`;

const ProgressFill = styled.div<{ $progress: number }>`
  height: 100%;
  background: linear-gradient(90deg, #9B59B6 0%, #B56FC1 100%);
  border-radius: inherit;
  width: ${({ $progress }) => $progress}%;
  transition: width 0.5s ease-in-out;
`;

const MeditationOptions = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
  flex-wrap: wrap;
`;

const MeditationOption = styled.button<{ $selected?: boolean }>`
  background: ${({ $selected }) => $selected ? '#9B59B6' : 'transparent'};
  color: ${({ $selected }) => $selected ? 'white' : '#9B59B6'};
  border: 1px solid #9B59B6;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: #9B59B6;
    color: white;
  }
`;

const MeditationControls = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  justify-content: center;
`;

const MeditationButton = styled.button<{ $variant: 'primary' | 'secondary' }>`
  background: ${({ $variant }) => $variant === 'primary' ? '#9B59B6' : 'transparent'};
  color: ${({ $variant }) => $variant === 'primary' ? 'white' : '#9B59B6'};
  border: 2px solid #9B59B6;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.mobile.xs} ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: ${({ $variant }) => $variant === 'primary' ? '0 4px 12px rgba(155, 89, 182, 0.3)' : 'none'};
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
`;

export const MeditationTimer: React.FC<MeditationTimerProps> = ({
  totalMinutes,
  sessionGoal,
  onStartSession,
  onCompleteSession,
  guidedOptions = [],
  isLoading = false,
  compact = false,
  className,
}) => {
  const [selectedOption, setSelectedOption] = useState<MeditationOption | null>(null);
  const [isMeditating, setIsMeditating] = useState(false);
  const [sessionTime, setSessionTime] = useState(0);

  const progress = sessionGoal > 0 ? Math.min((totalMinutes / sessionGoal) * 100, 100) : 0;

  const defaultOptions: MeditationOption[] = [
    { id: 'breathing-3', name: 'Breathing', duration: 3, type: 'BREATHING' },
    { id: 'mindfulness-5', name: 'Mindfulness', duration: 5, type: 'MINDFULNESS' },
    { id: 'guided-10', name: 'Guided', duration: 10, type: 'GUIDED' },
    { id: 'breathing-15', name: 'Deep Breathing', duration: 15, type: 'BREATHING' },
  ];

  const options = guidedOptions.length > 0 ? guidedOptions : defaultOptions;

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isMeditating) {
      interval = setInterval(() => {
        setSessionTime(prev => prev + 1);
      }, 60000); // Update every minute
    }
    return () => clearInterval(interval);
  }, [isMeditating]);

  const handleStartSession = () => {
    if (selectedOption) {
      setIsMeditating(true);
      setSessionTime(0);
      onStartSession(selectedOption.duration);
    }
  };

  const handleCompleteSession = () => {
    const duration = selectedOption?.duration || sessionTime;
    const quality = 4; // Default quality
    setIsMeditating(false);
    setSessionTime(0);
    onCompleteSession(duration, quality);
  };

  const formatTime = (minutes: number) => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  return (
    <MeditationContainer $compact={compact} className={className}>
      <MeditationTitle>🧘 Meditation</MeditationTitle>

      <MeditationStats>
        <span>
          <StatLabel>Total: </StatLabel>
          <StatValue $color="#9B59B6">{formatTime(totalMinutes)}</StatValue>
        </span>
        <span>
          <StatLabel>Goal: </StatLabel>
          <StatValue $color="#27AE60">{formatTime(sessionGoal)}</StatValue>
        </span>
      </MeditationStats>

      <ProgressBar>
        <ProgressFill $progress={progress} />
      </ProgressBar>

      {!compact && (
        <MeditationOptions>
          {options.map((option) => (
            <MeditationOption
              key={option.id}
              $selected={selectedOption?.id === option.id}
              onClick={() => setSelectedOption(option)}
              disabled={isMeditating}
            >
              {option.name} ({option.duration}m)
            </MeditationOption>
          ))}
        </MeditationOptions>
      )}

      {isMeditating && (
        <div style={{
          textAlign: 'center',
          fontSize: '12px',
          color: '#9B59B6',
          fontWeight: 'bold',
          marginBottom: '8px',
        }}>
          🧘 Meditating: {sessionTime}m {selectedOption && `- ${selectedOption.name}`}
        </div>
      )}

      <MeditationControls>
        {!isMeditating ? (
          <MeditationButton
            $variant="primary"
            onClick={handleStartSession}
            disabled={!selectedOption || isLoading}
          >
            🧘 Start {selectedOption ? `(${selectedOption.duration}m)` : '(Select option)'}
          </MeditationButton>
        ) : (
          <MeditationButton
            $variant="secondary"
            onClick={handleCompleteSession}
            disabled={isLoading}
          >
            ✅ Complete Session ({sessionTime}m)
          </MeditationButton>
        )}
      </MeditationControls>
    </MeditationContainer>
  );
};

export type { MeditationTimerProps, MeditationOption };