import React, { useState } from 'react';
import styled from 'styled-components';

interface MovementType {
  id: string;
  name: string;
  icon: string;
  intensity: 'LOW' | 'MEDIUM' | 'HIGH';
}

interface ActivityLoggerProps {
  movementTypes: MovementType[];
  onLog: (duration: number, type: string, intensity: string) => void;
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

const ActivityForm = styled.div`
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

const DurationSelector = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: ${({ theme }) => theme.spacing.sm};
`;

const DurationButton = styled.button<{ selected: boolean }>`
  padding: ${({ theme }) => theme.spacing.sm};
  border: 2px solid ${({ theme, selected }) => (selected ? theme.colors.sunriseOrange : theme.colors.neutral[200])};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  background: ${({ theme, selected }) => (selected ? theme.colors.sunriseOrange + '10' : 'white')};
  color: ${({ theme, selected }) => (selected ? theme.colors.sunriseOrange : theme.colors.neutral[600])};
  font-weight: ${({ theme, selected }) => (selected ? theme.typography.fontWeight.semibold : theme.typography.fontWeight.medium)};
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};

  &:hover {
    border-color: ${({ theme }) => theme.colors.sunriseOrange};
    background: ${({ theme }) => theme.colors.sunriseOrange + '05'};
  }
`;

const CustomDurationInput = styled.input`
  width: 100%;
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border: 1px solid ${({ theme }) => theme.colors.neutral[300]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  transition: border-color 0.2s ease;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.sunriseOrange};
    box-shadow: 0 0 0 3px rgba(230, 126, 80, 0.1);
  }
`;

const MovementTypeGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing.sm};
`;

const MovementTypeButton = styled.button<{ selected: boolean; intensity: string }>`
  padding: ${({ theme }) => theme.spacing.md};
  border: 2px solid ${({ theme, selected }) => (selected ? theme.colors.sunriseOrange : theme.colors.neutral[200])};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  background: ${({ theme, selected }) => (selected ? theme.colors.sunriseOrange + '10' : 'white')};
  color: ${({ theme, selected }) => (selected ? theme.colors.sunriseOrange : theme.colors.neutral[600])};
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  position: relative;

  ${({ intensity, theme }) => {
    const intensityColors = {
      LOW: theme.colors.success,
      MEDIUM: theme.colors.warning,
      HIGH: theme.colors.error,
    };
    return `
      border-left: 4px solid ${intensityColors[intensity]};
    `;
  }}

  &:hover {
    border-color: ${({ theme }) => theme.colors.sunriseOrange};
    background: ${({ theme }) => theme.colors.sunriseOrange + '05'};
    transform: translateY(-1px);
  }

  .activity-icon {
    font-size: 24px;
  }

  .activity-name {
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  }

  .intensity-badge {
    position: absolute;
    top: 4px;
    right: 4px;
    background: ${({ intensity, theme }) => {
      const colors = {
        LOW: theme.colors.success,
        MEDIUM: theme.colors.warning,
        HIGH: theme.colors.error,
      };
      return colors[intensity];
    }};
    color: white;
    border-radius: ${({ theme }) => theme.borderRadius.full};
    padding: 2px 6px;
    font-size: 10px;
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  }
`;

const IntensityLegend = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.md};
  justify-content: center;
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.neutral[500];
`;

const IntensityItem = styled.div<{ intensity: string }>`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: ${({ intensity, theme }) => {
      const colors = {
        LOW: theme.colors.success,
        MEDIUM: theme.colors.warning,
        HIGH: theme.colors.error,
      };
      return colors[intensity];
    }};
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
          background: ${theme.colors.sunriseOrange};
          color: white;
          border: 1px solid ${theme.colors.sunriseOrange};

          &:hover {
            background: ${theme.colors.sunriseOrange}dd;
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

export const ActivityLogger: React.FC<ActivityLoggerProps> = ({
  movementTypes,
  onLog,
  onClose,
}) => {
  const [selectedDuration, setSelectedDuration] = useState(5);
  const [customDuration, setCustomDuration] = useState('');
  const [selectedType, setSelectedType] = useState(movementTypes[0]?.id || '');
  const [isCustomDuration, setIsCustomDuration] = useState(false);

  const quickDurations = [5, 10, 15, 30];

  const selectedMovementType = movementTypes.find(type => type.id === selectedType);

  const handleDurationSelect = (duration: number) => {
    setSelectedDuration(duration);
    setIsCustomDuration(false);
    setCustomDuration('');
  };

  const handleCustomDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setCustomDuration(value);
    setIsCustomDuration(true);
  };

  const handleSubmit = () => {
    const finalDuration = isCustomDuration ? parseInt(customDuration) : selectedDuration;

    if (!finalDuration || finalDuration <= 0 || !selectedMovementType) {
      return;
    }

    onLog(finalDuration, selectedMovementType.name, selectedMovementType.intensity);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSubmit();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  return (
    <ModalOverlay onClick={onClose}>
      <ModalContent onClick={(e) => e.stopPropagation()} onKeyPress={handleKeyPress}>
        <ModalHeader>
          <h3>
            🏃‍♀️ Log Movement Activity
          </h3>
        </ModalHeader>

        <ActivityForm>
          <FormField>
            <Label>Duration</Label>
            <DurationSelector>
              {quickDurations.map((duration) => (
                <DurationButton
                  key={duration}
                  selected={!isCustomDuration && selectedDuration === duration}
                  onClick={() => handleDurationSelect(duration)}
                >
                  {duration} min
                </DurationButton>
              ))}
              <DurationButton
                selected={isCustomDuration}
                onClick={() => setIsCustomDuration(true)}
              >
                Custom
              </DurationButton>
            </DurationSelector>

            {isCustomDuration && (
              <CustomDurationInput
                type="number"
                min="1"
                max="120"
                value={customDuration}
                onChange={handleCustomDurationChange}
                placeholder="Enter minutes"
                aria-label="Custom duration in minutes"
              />
            )}
          </FormField>

          <FormField>
            <Label>Activity Type</Label>
            <MovementTypeGrid>
              {movementTypes.map((type) => (
                <MovementTypeButton
                  key={type.id}
                  selected={selectedType === type.id}
                  intensity={type.intensity}
                  onClick={() => setSelectedType(type.id)}
                >
                  <span className="activity-icon">{type.icon}</span>
                  <span className="activity-name">{type.name}</span>
                  <span className="intensity-badge">{type.intensity[0]}</span>
                </MovementTypeButton>
              ))}
            </MovementTypeGrid>

            <IntensityLegend>
              <IntensityItem intensity="LOW">
                <span className="dot"></span>
                <span>Low</span>
              </IntensityItem>
              <IntensityItem intensity="MEDIUM">
                <span className="dot"></span>
                <span>Medium</span>
              </IntensityItem>
              <IntensityItem intensity="HIGH">
                <span className="dot"></span>
                <span>High</span>
              </IntensityItem>
            </IntensityLegend>
          </FormField>

          <ActionButtons>
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" onClick={handleSubmit}>
              Log Activity
            </Button>
          </ActionButtons>
        </ActivityForm>
      </ModalContent>
    </ModalOverlay>
  );
};