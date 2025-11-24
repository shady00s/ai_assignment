import React from 'react';
import styled from 'styled-components';
import { MeditationOption } from '../../../../../types';

interface GuidedMeditationSelectorProps {
  options: MeditationOption[];
  onSelect: (option: MeditationOption) => void;
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

const MeditationGrid = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
`;

const MeditationOptionCard = styled.div<{ selected: boolean; type: string }>`
  padding: ${({ theme }) => theme.spacing.md};
  border: 2px solid ${({ theme, selected }) => (selected ? theme.colors.warning : theme.colors.neutral[200])};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  background: ${({ theme, selected }) => (selected ? theme.colors.warning + '10' : 'white')};
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.md};

  &:hover {
    border-color: ${({ theme }) => theme.colors.warning};
    background: ${({ theme }) => theme.colors.warning + '05'};
    transform: translateY(-1px);
  }

  .option-icon {
    font-size: 32px;
    flex-shrink: 0;
  }

  .option-content {
    flex: 1;
  }

  .option-title {
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    color: ${({ theme }) => theme.colors.neutral[700]};
    margin-bottom: 4px;
  }

  .option-details {
    display: flex;
    align-items: center;
    gap: ${({ theme }) => theme.spacing.sm};
    margin-bottom: 4px;
  }

  .option-duration {
    background: ${({ theme }) => theme.colors.warning};
    color: white;
    padding: 2px 8px;
    border-radius: ${({ theme }) => theme.borderRadius.full};
    font-size: ${({ theme }) => theme.typography.fontSize.xs};
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  }

  .option-type {
    background: ${({ theme, type }) => {
      switch (type) {
        case 'BREATHING': return theme.colors.success;
        case 'MINDFULNESS': return theme.colors.primary;
        case 'GUIDED': return theme.colors.warning;
        default: return theme.colors.neutral[500];
      }
    }};
    color: white;
    padding: 2px 8px;
    border-radius: ${({ theme }) => theme.borderRadius.full};
    font-size: ${({ theme }) => theme.typography.fontSize.xs};
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  }

  .option-description {
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    color: ${({ theme }) => theme.colors.neutral[500]};
    line-height: 1.4;
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
          background: ${theme.colors.warning};
          color: white;
          border: 1px solid ${theme.colors.warning};

          &:hover {
            background: ${theme.colors.warning}dd;
          }
        `;
      case 'secondary':
        return `
          background: white;
          color: ${theme.colors.neutral[600]};
          border: 1px solid ${theme.colors.neutral[300]};

          &:hover {
            background: ${theme.colors.neutral[50};
          }
        `;
    }
  }}
`;

const meditatiIconMap: Record<string, string> = {
  BREATHING: '🫁',
  MINDFULNESS: '🧘',
  GUIDED: '🎧',
};

export const GuidedMeditationSelector: React.FC<GuidedMeditationSelectorProps> = ({
  options,
  onSelect,
  onClose,
}) => {
  const [selectedOption, setSelectedOption] = React.useState<MeditationOption | null>(null);

  const handleOptionSelect = (option: MeditationOption) => {
    setSelectedOption(option);
  };

  const handleConfirm = () => {
    if (selectedOption) {
      onSelect(selectedOption);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleConfirm();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  return (
    <ModalOverlay onClick={onClose}>
      <ModalContent onClick={(e) => e.stopPropagation()} onKeyPress={handleKeyPress}>
        <ModalHeader>
          <h3>
            🎧 Choose Guided Meditation
          </h3>
        </ModalHeader>

        <MeditationGrid>
          {options.map((option) => (
            <MeditationOptionCard
              key={option.id}
              selected={selectedOption?.id === option.id}
              type={option.type}
              onClick={() => handleOptionSelect(option)}
            >
              <div className="option-icon">
                {meditatiIconMap[option.type] || '🧘'}
              </div>
              <div className="option-content">
                <div className="option-title">{option.name}</div>
                <div className="option-details">
                  <span className="option-duration">{option.duration} min</span>
                  <span className="option-type">{option.type}</span>
                </div>
                {option.description && (
                  <div className="option-description">{option.description}</div>
                )}
              </div>
            </MeditationOptionCard>
          ))}
        </MeditationGrid>

        <ActionButtons>
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleConfirm}
            disabled={!selectedOption}
          >
            Start Meditation
          </Button>
        </ActionButtons>
      </ModalContent>
    </ModalOverlay>
  );
};