import React, { useState } from 'react';
import styled from 'styled-components';

interface CheckInModalProps {
  currentMood: number;
  currentStress: number;
  currentEnergy: number;
  onCheckIn: (mood: number, stress: number, energy: number) => void;
  onClose: () => void;
}

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: ${({ theme }) => theme.spacing.md};
`;

const ModalContent = styled.div`
  background: white;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.lg};
  max-width: 500px;
  width: 100%;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
`;

const ModalHeader = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  text-align: center;

  h3 {
    margin: 0 0 ${({ theme }) => theme.spacing.sm} 0;
    color: ${({ theme }) => theme.colors.neutral[700]};
    font-size: ${({ theme }) => theme.typography.fontSize.xl};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  }

  .subtitle {
    color: ${({ theme }) => theme.colors.neutral[500]};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
  }
`;

const CheckInForm = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.xl};
`;

const MetricSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
`;

const MetricHeader = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.sm};
  margin-bottom: ${({ theme }) => theme.spacing.xs};

  h4 {
    margin: 0;
    color: ${({ theme }) => theme.colors.neutral[700]};
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  }

  .metric-icon {
    font-size: 24px;
  }
`;

const MoodScale = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${({ theme }) => theme.spacing.md};
  background: ${({ theme }) => theme.colors.neutral[50]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border: 2px solid ${({ theme }) => theme.colors.neutral[200]};
`;

const ScaleButton = styled.button<{ selected: boolean; value: number }>`
  background: none;
  border: none;
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  transition: all 0.2s ease;
  position: relative;
  flex: 1;

  .emoji {
    font-size: 32px;
    display: block;
    margin-bottom: 4px;
    transform: ${({ selected }) => selected ? 'scale(1.2)' : 'scale(1)'};
    transition: transform 0.2s ease;
  }

  .label {
    font-size: ${({ theme }) => theme.typography.fontSize.xs};
    color: ${({ theme, selected }) => selected ? theme.colors.primary : theme.colors.neutral[500]};
    font-weight: ${({ theme, selected }) => selected ? theme.typography.fontWeight.semibold : theme.typography.fontWeight.medium};
  }

  &:hover {
    background: ${({ theme }) => theme.colors.neutral[100]};
  }

  &:hover .emoji {
    transform: scale(1.1);
  }

  ${({ selected, theme, value }) => selected && `
    background: ${theme.colors.primary}10;
    border: 1px solid ${theme.colors.primary}30;

    .emoji {
      animation: bounce 0.3s ease;
    }
  `}

  @keyframes bounce {
    0%, 100% { transform: scale(1.2); }
    50% { transform: scale(1.3); }
  }
`;

const NotesSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const NotesTextarea = styled.textarea`
  width: 100%;
  min-height: 80px;
  padding: ${({ theme }) => theme.spacing.sm};
  border: 1px solid ${({ theme }) => theme.colors.neutral[300]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  resize: vertical;
  font-family: inherit;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary};
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
  }

  &::placeholder {
    color: ${({ theme }) => theme.colors.neutral[400]};
  }
`;

const ActionButtons = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.md};
  margin-top: ${({ theme }) => theme.spacing.lg};
`;

const Button = styled.button<{ variant?: 'primary' | 'secondary' }>`
  flex: 1;
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all 0.2s ease;

  ${({ variant = 'primary', theme }) => {
    switch (variant) {
      case 'primary':
        return `
          background: ${theme.colors.primary};
          color: white;
          border: 1px solid ${theme.colors.primary};

          &:hover {
            background: ${theme.colors.primary}dd;
          }
        `;
      case 'secondary':
        return `
          background: white;
          color: ${theme.colors.neutral[600]};
          border: 1px solid ${theme.colors.neutral[300]};

          &:hover {
            background: ${theme.colors.neutral[50]};
          }
        `;
    }
  }}
`;

const WellnessSummary = styled.div`
  padding: ${({ theme }) => theme.spacing.md};
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.primary}10, ${({ theme }) => theme.colors.success}10);
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border-left: 4px solid ${({ theme }) => theme.colors.primary};
  margin-top: ${({ theme }) => theme.spacing.lg};
`;

const SummaryTitle = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.primary};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const SummaryStats = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing.sm};
  text-align: center;

  .stat-value {
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
    color: ${({ theme }) => theme.colors.neutral[700]};
  }

  .stat-label {
    font-size: ${({ theme }) => theme.typography.fontSize.xs};
    color: ${({ theme }) => theme.colors.neutral[500]};
    margin-top: 2px;
  }
`;

const metricConfig = {
  mood: {
    title: 'Mood',
    icon: '😊',
    values: [
      { emoji: '😢', label: 'Very Low', value: 1 },
      { emoji: '😕', label: 'Low', value: 2 },
      { emoji: '😐', label: 'Neutral', value: 3 },
      { emoji: '🙂', label: 'Good', value: 4 },
      { emoji: '😊', label: 'Excellent', value: 5 },
    ],
  },
  stress: {
    title: 'Stress',
    icon: '😰',
    values: [
      { emoji: '😌', label: 'Very Relaxed', value: 1 },
      { emoji: '😊', label: 'Relaxed', value: 2 },
      { emoji: '😐', label: 'Moderate', value: 3 },
      { emoji: '😰', label: 'Stressed', value: 4 },
      { emoji: '😫', label: 'Very Stressed', value: 5 },
    ],
  },
  energy: {
    title: 'Energy',
    icon: '⚡',
    values: [
      { emoji: '😴', label: 'Very Low', value: 1 },
      { emoji: '🔋', label: 'Low', value: 2 },
      { emoji: '⚡', label: 'Moderate', value: 3 },
      { emoji: '🚀', label: 'High', value: 4 },
      { emoji: '🔥', label: 'Very High', value: 5 },
    ],
  },
};

export const CheckInModal: React.FC<CheckInModalProps> = ({
  currentMood,
  currentStress,
  currentEnergy,
  onCheckIn,
  onClose,
}) => {
  const [mood, setMood] = useState(currentMood);
  const [stress, setStress] = useState(currentStress);
  const [energy, setEnergy] = useState(currentEnergy);
  const [notes, setNotes] = useState('');

  const handleSubmit = () => {
    onCheckIn(mood, stress, energy);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleSubmit();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  const calculateOverallWellness = (): number => {
    const adjustedStress = 6 - stress; // Invert stress (lower is better)
    return Math.round((mood + adjustedStress + energy) / 3);
  };

  const overallWellness = calculateOverallWellness();

  return (
    <ModalOverlay onClick={onClose}>
      <ModalContent onClick={(e) => e.stopPropagation()} onKeyPress={handleKeyPress}>
        <ModalHeader>
          <h3>📝 Daily Wellness Check-In</h3>
          <div className="subtitle">How are you feeling today?</div>
        </ModalHeader>

        <CheckInForm>
          {Object.entries(metricConfig).map(([key, config]) => {
            const currentValue = key === 'mood' ? mood : key === 'stress' ? stress : energy;
            const setCurrentValue = key === 'mood' ? setMood : key === 'stress' ? setStress : setEnergy;

            return (
              <MetricSection key={key}>
                <MetricHeader>
                  <span className="metric-icon">{config.icon}</span>
                  <h4>{config.title}</h4>
                </MetricHeader>
                <MoodScale>
                  {config.values.map((option) => (
                    <ScaleButton
                      key={option.value}
                      selected={currentValue === option.value}
                      value={option.value}
                      onClick={() => setCurrentValue(option.value)}
                      aria-label={`Set ${config.title.toLowerCase()} to ${option.label}`}
                      aria-pressed={currentValue === option.value}
                    >
                      <span className="emoji" role="img" aria-label={option.label}>
                        {option.emoji}
                      </span>
                      <span className="label">{option.label}</span>
                    </ScaleButton>
                  ))}
                </MoodScale>
              </MetricSection>
            );
          })}

          <NotesSection>
            <label htmlFor="wellness-notes">
              <strong>Notes (Optional)</strong>
            </label>
            <NotesTextarea
              id="wellness-notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Any thoughts about your day, what's influencing your mood, or goals for tomorrow..."
            />
          </NotesSection>

          <WellnessSummary>
            <SummaryTitle>Today's Wellness Score</SummaryTitle>
            <SummaryStats>
              <div>
                <div className="stat-value">{overallWellness}/5</div>
                <div className="stat-label">Overall</div>
              </div>
              <div>
                <div className="stat-value">{mood}/5</div>
                <div className="stat-label">Mood</div>
              </div>
              <div>
                <div className="stat-value">{energy}/5</div>
                <div className="stat-label">Energy</div>
              </div>
            </SummaryStats>
          </WellnessSummary>

          <ActionButtons>
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" onClick={handleSubmit}>
              Save Check-In
            </Button>
          </ActionButtons>
        </CheckInForm>
      </ModalContent>
    </ModalOverlay>
  );
};