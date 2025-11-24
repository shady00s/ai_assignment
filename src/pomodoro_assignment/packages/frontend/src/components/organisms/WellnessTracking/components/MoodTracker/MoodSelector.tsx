import React from 'react';
import styled from 'styled-components';

interface MoodSelectorProps {
  metric: 'mood' | 'stress' | 'energy';
  currentValue: number;
  onSelect: (value: number) => void;
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
  max-width: 400px;
  width: 100%;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
`;

const ModalHeader = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.lg};

  h3 {
    margin: 0;
    color: ${({ theme }) => theme.colors.neutral[700]};
    font-size: ${({ theme }) => theme.typography.fontSize.xl};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    display: flex;
    align-items: center;
    gap: ${({ theme }) => theme.spacing.sm};
  }
`;

const MoodGrid = styled.div`
  display: flex;
  justify-content: space-around;
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  padding: ${({ theme }) => theme.spacing.md};
  background: ${({ theme }) => theme.colors.neutral[50]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
`;

const MoodOption = styled.button<{ selected: boolean }>`
  background: none;
  border: none;
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.sm};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  transition: all 0.2s ease;
  position: relative;

  .mood-emoji {
    font-size: 48px;
    display: block;
    margin-bottom: ${({ theme }) => theme.spacing.xs};
    transform: ${({ selected }) => selected ? 'scale(1.2)' : 'scale(1)'};
    transition: transform 0.2s ease;
  }

  .mood-label {
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    color: ${({ theme, selected }) => selected ? theme.colors.primary : theme.colors.neutral[600]};
    font-weight: ${({ theme, selected }) => selected ? theme.typography.fontWeight.semibold : theme.typography.fontWeight.medium};
  }

  &:hover {
    background: ${({ theme }) => theme.colors.neutral[100]};
  }

  &:hover .mood-emoji {
    transform: scale(1.1);
  }

  ${({ selected, theme }) => selected && `
    .mood-emoji {
      animation: bounce 0.5s ease;
    }
  `}

  @keyframes bounce {
    0%, 100% { transform: scale(1.2); }
    50% { transform: scale(1.4); }
  }
`;

const ActionButtons = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.md};
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

const metricConfig = {
  mood: {
    title: 'How are you feeling?',
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
    title: 'What\'s your stress level?',
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
    title: 'How\'s your energy level?',
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

export const MoodSelector: React.FC<MoodSelectorProps> = ({
  metric,
  currentValue,
  onSelect,
  onClose,
}) => {
  const config = metricConfig[metric];

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  return (
    <ModalOverlay onClick={onClose}>
      <ModalContent onClick={(e) => e.stopPropagation()} onKeyPress={handleKeyPress}>
        <ModalHeader>
          <h3>
            {config.icon} {config.title}
          </h3>
        </ModalHeader>

        <MoodGrid>
          {config.values.map((option) => (
            <MoodOption
              key={option.value}
              selected={currentValue === option.value}
              onClick={() => onSelect(option.value)}
              aria-label={`Select ${option.label}`}
              aria-pressed={currentValue === option.value}
            >
              <span className="mood-emoji" role="img" aria-label={option.label}>
                {option.emoji}
              </span>
              <span className="mood-label">{option.label}</span>
            </MoodOption>
          ))}
        </MoodGrid>

        <ActionButtons>
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={() => onSelect(currentValue)}
          >
            Confirm
          </Button>
        </ActionButtons>
      </ModalContent>
    </ModalOverlay>
  );
};