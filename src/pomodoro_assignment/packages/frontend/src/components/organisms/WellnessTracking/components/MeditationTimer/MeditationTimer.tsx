import React, { useState, useCallback, useEffect } from 'react';
import styled from 'styled-components';
import { MeditationTimerProps, MeditationOption } from '../../../../../types';
import { Card } from '../../../atoms';
import { BreathingVisualizer } from './BreathingVisualizer';
import { SessionControls } from './SessionControls';
import { GuidedMeditationSelector } from './GuidedMeditationSelector';

const MeditationContainer = styled(Card)<{ compact?: boolean }>`
  padding: ${({ theme, compact }) => compact ? theme.spacing.md : theme.spacing.lg};
  height: fit-content;
  position: relative;
  background: ${({ theme, isActive }) =>
    isActive ?
    `linear-gradient(135deg, ${theme.colors.sageGreen}10, ${theme.colors.primary}10)` :
    'white'};
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
    display: flex;
    align-items: center;
    gap: ${({ theme }) => theme.spacing.sm};
  }

  .meditation-icon {
    font-size: 24px;
    animation: float 3s ease-in-out infinite;
  }

  @keyframes float {
    0%, 100% {
      transform: translateY(0px);
    }
    50% {
      transform: translateY(-5px);
    }
  }
`;

const MainContent = styled.div<{ compact?: boolean }>`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.lg};
`;

const TimerDisplay = styled.div<{ compact?: boolean }>`
  font-size: ${({ compact }) => compact ? '32px' : '48px'};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.sageGreen};
  font-variant-numeric: tabular-nums;
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing.md};
  min-width: 120px;
`;

const SessionInfo = styled.div`
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing.md};

  .session-type {
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    color: ${({ theme }) => theme.colors.neutral[700]};
    margin-bottom: 4px;
  }

  .session-description {
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    color: ${({ theme }) => theme.colors.neutral[500]};
    line-height: 1.4;
  }
`;

const ProgressSection = styled.div<{ compact?: boolean }>`
  width: ${({ compact }) => compact ? '200px' : '250px'};
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const ProgressBar = styled.div<{ compact?: boolean }>`
  width: 100%;
  height: 8px;
  background-color: ${({ theme }) => theme.colors.neutral[200]};
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const ProgressFill = styled.div<{ progress: number }>`
  height: 100%;
  background: linear-gradient(90deg, ${({ theme }) => theme.colors.sageGreen}, ${({ theme }) => theme.colors.primary});
  border-radius: 4px;
  width: ${({ progress }) => progress}%;
  transition: width 1s linear;
`;

const ProgressText = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.neutral[500]};
`;

const StatsGrid = styled.div<{ compact?: boolean }>`
  display: grid;
  grid-template-columns: ${({ compact }) => compact ? '1fr' : '1fr 1fr 1fr'};
  gap: ${({ theme }) => theme.spacing.md};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const StatCard = styled.div<{ variant?: 'primary' | 'secondary' | 'accent' }>`
  padding: ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  ${({ variant = 'primary', theme }) => {
    switch (variant) {
      case 'primary':
        return `
          background: ${theme.colors.sageGreen}10;
          border: 1px solid ${theme.colors.sageGreen}30;
        `;
      case 'secondary':
        return `
          background: ${theme.colors.primary}10;
          border: 1px solid ${theme.colors.primary}30;
        `;
      case 'accent':
        return `
          background: ${theme.colors.warning}10;
          border: 1px solid ${theme.colors.warning}30;
        `;
    }
  }}

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
`;

const StatIcon = styled.div`
  font-size: 20px;
  margin-bottom: 4px;
`;

const StatValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.neutral[700]};
  margin-bottom: 2px;
`;

const StatLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const QuickStartGrid = styled.div<{ compact?: boolean }>`
  display: grid;
  grid-template-columns: ${({ compact }) => compact ? '1fr' : 'repeat(3, 1fr)'};
  gap: ${({ theme }) => theme.spacing.sm};
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const QuickStartButton = styled.button<{ variant?: 'breathing' | 'mindfulness' | 'guided' }>`
  padding: ${({ theme }) => theme.spacing.sm};
  border: 2px solid ${({ theme, variant }) => {
    switch (variant) {
      case 'breathing': return theme.colors.success;
      case 'mindfulness': return theme.colors.primary;
      case 'guided': return theme.colors.warning;
      default: return theme.colors.neutral[300];
    }
  };
  border-radius: ${({ theme }) => theme.borderRadius.md};
  background: white;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};

  &:hover {
    background: ${({ theme, variant }) => {
      switch (variant) {
        case 'breathing': return theme.colors.success}10;
        case 'mindfulness': return theme.colors.primary}10;
        case 'guided': return theme.colors.warning}10;
        default: return theme.colors.neutral[50];
      }
    };
    transform: translateY(-1px);
  }

  .button-icon {
    font-size: 24px;
  }

  .button-text {
    font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
    color: ${({ theme }) => theme.colors.neutral[700]};
  }
`;

const AchievementBadge = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.sageGreen}, ${({ theme }) => theme.colors.primary});
  color: white;
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  margin-top: ${({ theme }) => theme.spacing.md};
`;

const defaultMeditationOptions: MeditationOption[] = [
  {
    id: '1',
    name: '5 min Breathing',
    duration: 5,
    type: 'BREATHING',
    description: 'Quick breathing exercise for relaxation',
  },
  {
    id: '2',
    name: '10 min Mindfulness',
    duration: 10,
    type: 'MINDFULNESS',
    description: 'Basic mindfulness practice',
  },
  {
    id: '3',
    name: '15 min Guided',
    duration: 15,
    type: 'GUIDED',
    description: 'Guided meditation for focus',
  },
  {
    id: '4',
    name: '20 min Deep Focus',
    duration: 20,
    type: 'MINDFULNESS',
    description: 'Extended meditation session',
  },
];

export const MeditationTimer: React.FC<MeditationTimerProps> = ({
  totalMinutes,
  sessionGoal = 15,
  onStartSession,
  onCompleteSession,
  guidedOptions = defaultMeditationOptions,
  isLoading,
  compact = false,
}) => {
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [currentSession, setCurrentSession] = useState<{ minutes: number; type: string } | null>(null);
  const [remainingSeconds, setRemainingSeconds] = useState(0);
  const [showGuidedSelector, setShowGuidedSelector] = useState(false);
  const [breathingPhase, setBreathingPhase] = useState<'inhale' | 'hold' | 'exhale'>('inhale');

  const progressPercentage = sessionGoal > 0 ? Math.min(100, Math.round((totalMinutes / sessionGoal) * 100)) : 0;
  const isGoalReached = totalMinutes >= sessionGoal;

  const formatTime = (totalSeconds: number): string => {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handleQuickStart = useCallback((type: 'breathing' | 'mindfulness' | 'guided', duration: number) => {
    if (isLoading) return;

    const sessionType = type === 'breathing' ? 'BREATHING' : type === 'mindfulness' ? 'MINDFULNESS' : 'GUIDED';
    setCurrentSession({ minutes: duration, type: sessionType });
    setRemainingSeconds(duration * 60);
    setIsSessionActive(true);
    onStartSession(duration);
  }, [isLoading, onStartSession]);

  const handleGuidedSelect = useCallback((option: MeditationOption) => {
    if (isLoading) return;

    setCurrentSession({ minutes: option.duration, type: option.type });
    setRemainingSeconds(option.duration * 60);
    setIsSessionActive(true);
    setShowGuidedSelector(false);
    onStartSession(option.duration);
  }, [isLoading, onStartSession]);

  const handleComplete = useCallback((quality: number) => {
    if (!currentSession) return;

    setIsSessionActive(false);
    setCurrentSession(null);
    setRemainingSeconds(0);
    onCompleteSession(currentSession.minutes, quality);
  }, [currentSession, onCompleteSession]);

  const handleCancel = useCallback(() => {
    setIsSessionActive(false);
    setCurrentSession(null);
    setRemainingSeconds(0);
  }, []);

  // Breathing exercise phase cycling
  useEffect(() => {
    if (!isSessionActive || currentSession?.type !== 'BREATHING') return;

    const interval = setInterval(() => {
      setBreathingPhase((prev) => {
        const sequence: ('inhale' | 'hold' | 'exhale')[] = ['inhale', 'hold', 'exhale', 'hold'];
        const currentIndex = sequence.indexOf(prev);
        return sequence[(currentIndex + 1) % sequence.length];
      });
    }, 4000); // 4 seconds per phase

    return () => clearInterval(interval);
  }, [isSessionActive, currentSession]);

  // Timer countdown
  useEffect(() => {
    if (!isSessionActive || remainingSeconds <= 0) {
      if (remainingSeconds <= 0 && isSessionActive) {
        handleComplete(3); // Default quality rating
      }
      return;
    }

    const interval = setInterval(() => {
      setRemainingSeconds((prev) => {
        if (prev <= 1) {
          handleComplete(3);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isSessionActive, remainingSeconds, handleComplete]);

  const getSessionTypeDisplay = (): string => {
    if (!currentSession) return 'Ready to meditate';
    switch (currentSession.type) {
      case 'BREATHING':
        return 'Breathing Exercise';
      case 'MINDFULNESS':
        return 'Mindfulness';
      case 'GUIDED':
        return 'Guided Meditation';
      default:
        return 'Meditation';
    }
  };

  const getSessionDescription = (): string => {
    if (!currentSession) return 'Choose a meditation session below';
    switch (currentSession.type) {
      case 'BREATHING':
        return 'Focus on your breath and find calm';
      case 'MINDFULNESS':
        return 'Be present and aware of the moment';
      case 'GUIDED':
        return 'Follow along with guided instructions';
      default:
        return 'Time for mindfulness';
    }
  };

  return (
    <>
      <MeditationContainer compact={compact} isActive={isSessionActive}>
        <CardHeader>
          <h3>
            <span className="meditation-icon">🧘</span>
            Meditation Timer
          </h3>
          {!compact && totalMinutes > 0 && (
            <AchievementBadge>
              🏆 {totalMinutes} min today
            </AchievementBadge>
          )}
        </CardHeader>

        <MainContent compact={compact}>
          {isSessionActive ? (
            <>
              <TimerDisplay compact={compact}>
                {formatTime(remainingSeconds)}
              </TimerDisplay>

              <SessionInfo>
                <div className="session-type">{getSessionTypeDisplay()}</div>
                <div className="session-description">{getSessionDescription()}</div>
              </SessionInfo>

              {currentSession?.type === 'BREATHING' && (
                <BreathingVisualizer
                  phase={breathingPhase}
                  compact={compact}
                />
              )}

              <ProgressSection compact={compact}>
                <ProgressBar compact={compact}>
                  <ProgressFill progress={((currentSession?.minutes * 60 - remainingSeconds) / (currentSession?.minutes * 60)) * 100} />
                </ProgressBar>
                <ProgressText>
                  <span>{formatTime(remainingSeconds)} left</span>
                  <span>{formatTime(currentSession?.minutes * 60 || 0)} total</span>
                </ProgressText>
              </ProgressSection>

              <SessionControls
                isActive={isSessionActive}
                remainingSeconds={remainingSeconds}
                totalSeconds={currentSession?.minutes * 60 || 0}
                onComplete={handleComplete}
                onCancel={handleCancel}
                compact={compact}
              />
            </>
          ) : (
            <>
              <TimerDisplay compact={compact}>
                {totalMinutes}min
              </TimerDisplay>

              <SessionInfo>
                <div className="session-type">Today's Progress</div>
                <div className="session-description">{totalMinutes} of {sessionGoal} minutes completed</div>
              </SessionInfo>

              <ProgressSection compact={compact}>
                <ProgressBar compact={compact}>
                  <ProgressFill progress={progressPercentage} />
                </ProgressBar>
                <ProgressText>
                  <span>{totalMinutes} min</span>
                  <span>{sessionGoal} min goal</span>
                </ProgressText>
              </ProgressSection>

              <StatsGrid compact={compact}>
                <StatCard variant="primary">
                  <StatIcon>🧘</StatIcon>
                  <StatValue>{totalMinutes}</StatValue>
                  <StatLabel>Total Minutes</StatLabel>
                </StatCard>

                <StatCard variant="secondary">
                  <StatIcon>🎯</StatIcon>
                  <StatValue>{sessionGoal}</StatValue>
                  <StatLabel>Daily Goal</StatLabel>
                </StatCard>

                <StatCard variant="accent">
                  <StatIcon>📈</StatValue>
                  <StatValue>{progressPercentage}%</StatValue>
                  <StatLabel>Progress</StatLabel>
                </StatCard>
              </StatsGrid>

              <QuickStartGrid compact={compact}>
                <QuickStartButton
                  variant="breathing"
                  onClick={() => handleQuickStart('breathing', 5)}
                  disabled={isLoading}
                >
                  <span className="button-icon">🫁</span>
                  <span className="button-text">Breathing</span>
                </QuickStartButton>

                <QuickStartButton
                  variant="mindfulness"
                  onClick={() => handleQuickStart('mindfulness', 10)}
                  disabled={isLoading}
                >
                  <span className="button-icon">🧘</span>
                  <span className="button-text">Mindfulness</span>
                </QuickStartButton>

                <QuickStartButton
                  variant="guided"
                  onClick={() => setShowGuidedSelector(true)}
                  disabled={isLoading}
                >
                  <span className="button-icon">🎧</span>
                  <span className="button-text">Guided</span>
                </QuickStartButton>
              </QuickStartGrid>

              {isGoalReached && (
                <AchievementBadge>
                  🎉 Daily Goal Achieved!
                </AchievementBadge>
              )}
            </>
          )}
        </MainContent>
      </MeditationContainer>

      {showGuidedSelector && (
        <GuidedMeditationSelector
          options={guidedOptions}
          onSelect={handleGuidedSelect}
          onClose={() => setShowGuidedSelector(false)}
        />
      )}
    </>
  );
};