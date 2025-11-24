import React, { useState } from 'react';
import styled from 'styled-components';

interface GoalSettingsProps {
  currentGoal: number;
  onSave: (newGoal: number) => void;
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

const SettingsForm = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
`;

const FormField = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const Label = styled.label`
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.neutral[600]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
`;

const InputGroup = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.md};
`;

const NumberInput = styled.input`
  flex: 1;
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border: 1px solid ${({ theme }) => theme.colors.neutral[300]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  transition: border-color 0.2s ease;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary};
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
  }
`;

const UnitDisplay = styled.span`
  color: ${({ theme }) => theme.colors.neutral[500]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  min-width: 80px;
`;

const QuickSelectGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing.sm};
`;

const QuickSelectButton = styled.button<{ selected: boolean }>`
  padding: ${({ theme }) => theme.spacing.sm};
  border: 2px solid ${({ theme, selected }) => (selected ? theme.colors.primary : theme.colors.neutral[200])};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  background: ${({ theme, selected }) => (selected ? theme.colors.primary + '10' : 'white')};
  color: ${({ theme, selected }) => (selected ? theme.colors.primary : theme.colors.neutral[600])};
  font-weight: ${({ theme, selected }) => (selected ? theme.typography.fontWeight.semibold : theme.typography.fontWeight.medium)};
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary};
    background: ${({ theme }) => theme.colors.primary + '05'};
  }
`;

const SliderContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const Slider = styled.input`
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: ${({ theme }) => theme.colors.neutral[200]};
  outline: none;
  -webkit-appearance: none;

  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: ${({ theme }) => theme.colors.primary};
    cursor: pointer;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease;

    &:hover {
      transform: scale(1.1);
    }
  }

  &::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: ${({ theme }) => theme.colors.primary};
    cursor: pointer;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    border: none;
    transition: transform 0.2s ease;

    &:hover {
      transform: scale(1.1);
    }
  }
`;

const SliderValue = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.neutral[600]};
`;

const InfoMessage = styled.div`
  padding: ${({ theme }) => theme.spacing.sm};
  background: ${({ theme }) => theme.colors.info + '10'};
  border: 1px solid ${({ theme }) => theme.colors.info + '30'};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  color: ${({ theme }) => theme.colors.info};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  line-height: 1.4;
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

export const GoalSettings: React.FC<GoalSettingsProps> = ({
  currentGoal,
  onSave,
  onClose,
}) => {
  const [goalValue, setGoalValue] = useState(currentGoal.toString());
  const [isValid, setIsValid] = useState(true);

  const quickSelectOptions = [6, 8, 10, 12, 14, 16];

  const validateGoal = (value: string): boolean => {
    const num = parseInt(value);
    return !isNaN(num) && num >= 1 && num <= 20;
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setGoalValue(value);
    setIsValid(validateGoal(value));
  };

  const handleQuickSelect = (value: number) => {
    setGoalValue(value.toString());
    setIsValid(true);
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setGoalValue(value);
    setIsValid(true);
  };

  const handleSave = () => {
    const num = parseInt(goalValue);
    if (isValid && !isNaN(num)) {
      onSave(num);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  return (
    <ModalOverlay onClick={onClose}>
      <ModalContent onClick={(e) => e.stopPropagation()} onKeyPress={handleKeyPress}>
        <ModalHeader>
          <h3>
            💧 Hydration Goal Settings
          </h3>
        </ModalHeader>

        <SettingsForm>
          <FormField>
            <Label>Daily Glasses Goal</Label>
            <InputGroup>
              <NumberInput
                type="number"
                min="1"
                max="20"
                value={goalValue}
                onChange={handleInputChange}
                aria-label="Daily glasses goal"
              />
              <UnitDisplay>glasses/day</UnitDisplay>
            </InputGroup>
          </FormField>

          <FormField>
            <Label>Quick Select</Label>
            <QuickSelectGrid>
              {quickSelectOptions.map((option) => (
                <QuickSelectButton
                  key={option}
                  selected={parseInt(goalValue) === option}
                  onClick={() => handleQuickSelect(option)}
                >
                  {option}
                </QuickSelectButton>
              ))}
            </QuickSelectGrid>
          </FormField>

          <FormField>
            <Label>Adjust with Slider</Label>
            <SliderContainer>
              <Slider
                type="range"
                min="1"
                max="20"
                value={parseInt(goalValue) || 8}
                onChange={handleSliderChange}
              />
              <SliderValue>
                <span>1</span>
                <span>{goalValue} glasses</span>
                <span>20</span>
              </SliderValue>
            </SliderContainer>
          </FormField>

          <InfoMessage>
            💡 <strong>Health Tip:</strong> The recommended daily water intake is about 8 glasses (64oz or 2 liters), but individual needs may vary based on activity level, climate, and personal health factors.
          </InfoMessage>

          <ActionButtons>
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" onClick={handleSave} disabled={!isValid}>
              Save Goal
            </Button>
          </ActionButtons>
        </SettingsForm>
      </ModalContent>
    </ModalOverlay>
  );
};