import React, { useState } from 'react';
import styled from 'styled-components';

interface SessionControlsProps {
  isActive: boolean;
  remainingSeconds: number;
  totalSeconds: number;
  onComplete: (quality: number) => void;
  onCancel: () => void;
  compact?: boolean;
}

const ControlsContainer = styled.div<{ compact?: boolean }>`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
  align-items: center;
  width: 100%;
`;

const QualitySelector = styled.div<{ compact?: boolean }>`
  display: flex;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.sm};
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const QualityButton = styled.button<{ selected: boolean; quality: number }>`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: 2px solid ${({ theme, selected, quality }) => {
    if (selected) return theme.colors.success;
    return quality >= 4 ? theme.colors.success :
           quality >= 3 ? theme.colors.warning :
           theme.colors.error;
  }};
  background: white;
  color: ${({ theme, selected, quality }) => {
    if (selected) return theme.colors.success;
    return quality >= 4 ? theme.colors.success :
           quality >= 3 ? theme.colors.warning :
           theme.colors.error;
  }};
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  font-weight: bold;

  &:hover {
    transform: scale(1.1);
    background: ${({ theme, quality }) => {
      if (quality >= 4) return theme.colors.success}10;
      if (quality >= 3) return theme.colors.warning}10;
      return theme.colors.error}10;
    };
  }
`;

const ActionButtons = styled.div<{ compact?: boolean }>`
  display: flex;
  gap: ${({ theme }) => theme.spacing.md};
  justify-content: center;
  width: 100%;
`;

const ActionButton = styled.button<{ variant?: 'complete' | 'cancel' }>`
  padding: ${({ theme, compact }) => compact ? theme.spacing.sm : theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};

  ${({ variant = 'complete', theme }) => {
    switch (variant) {
      case 'complete':
        return `
          background: ${theme.colors.success};
          color: white;
          border: 1px solid ${theme.colors.success};

          &:hover {
            background: ${theme.colors.success}dd;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          }
        `;
      case 'cancel':
        return `
          background: ${theme.colors.error};
          color: white;
          border: 1px solid ${theme.colors.error};

          &:hover {
            background: ${theme.colors.error}dd;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          }
        `;
    }
  }}

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none !important;
  }
`;

const SessionProgress = styled.div<{ compact?: boolean }>`
  width: 100%;
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const ProgressText = styled.div<{ compact?: boolean }>`
  font-size: ${({ theme, compact }) => compact ? theme.typography.fontSize.sm : theme.typography.fontSize.base};
  color: ${({ theme }) => theme.colors.neutral[600]};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const CompletionMessage = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  color: ${({ theme }) => theme.colors.success};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

export const SessionControls: React.FC<SessionControlsProps> = ({
  isActive,
  remainingSeconds,
  totalSeconds,
  onComplete,
  onCancel,
  compact = false,
}) => {
  const [selectedQuality, setSelectedQuality] = useState(3);
  const [isCompleted, setIsCompleted] = useState(false);

  const isSessionFinished = remainingSeconds <= 0;
  const progress = totalSeconds > 0 ? Math.round(((totalSeconds - remainingSeconds) / totalSeconds) * 100) : 0;

  const handleComplete = () => {
    setIsCompleted(true);
    onComplete(selectedQuality);
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getQualityEmoji = (quality: number): string => {
    switch (quality) {
      case 1: return '😞';
      case 2: return '😐';
      case 3: return '🙂';
      case 4: return '😊';
      case 5: return '🤗';
      default: return '🙂';
    }
  };

  const getQualityLabel = (quality: number): string => {
    switch (quality) {
      case 1: return 'Poor';
      case 2: return 'Fair';
      case 3: return 'Good';
      case 4: return 'Great';
      case 5: return 'Excellent';
      default: return 'Good';
    }
  };

  if (!isActive) {
    return null;
  }

  if (isSessionFinished) {
    return (
      <ControlsContainer compact={compact}>
        <CompletionMessage>
          🎉 Session Complete!
        </CompletionMessage>

        <QualitySelector compact={compact}>
          {[1, 2, 3, 4, 5].map((quality) => (
            <QualityButton
              key={quality}
              selected={selectedQuality === quality}
              quality={quality}
              onClick={() => setSelectedQuality(quality)}
              aria-label={`Rate session ${getQualityLabel(quality)}`}
              title={getQualityLabel(quality)}
            >
              {getQualityEmoji(quality)}
            </QualityButton>
          ))}
        </QualitySelector>

        <ActionButtons compact={compact}>
          <ActionButton variant="complete" onClick={handleComplete}>
            ✅ Log Session ({getQualityLabel(selectedQuality)})
          </ActionButton>
        </ActionButtons>
      </ControlsContainer>
    );
  }

  return (
    <ControlsContainer compact={compact}>
      <SessionProgress compact={compact}>
        <ProgressText compact={compact}>
          Progress: {progress}% • {formatTime(remainingSeconds)} remaining
        </ProgressText>
      </SessionProgress>

      <ActionButtons compact={compact}>
        <ActionButton variant="cancel" onClick={onCancel}>
          ⏹️ End Session
        </ActionButton>
      </ActionButtons>
    </ControlsContainer>
  );
};