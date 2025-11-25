import React from 'react';
import styled from 'styled-components';

interface EnergyIndicatorProps {
  energyLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  score?: number; // 0-100
  showPercentage?: boolean;
  size?: 'small' | 'medium' | 'large';
  className?: string;
}

const EnergyContainer = styled.div<{ $size: 'small' | 'medium' | 'large' }>`
  display: flex;
  align-items: center;
  gap: ${({ theme, $size }) => {
    switch ($size) {
      case 'small': return theme.spacing.mobile.xs;
      case 'medium': return theme.spacing.mobile.sm;
      case 'large': return theme.spacing.mobile.md;
      default: return theme.spacing.mobile.sm;
    }
  }};
  padding: ${({ theme, $size }) => {
    switch ($size) {
      case 'small': return `${theme.spacing.mobile.xs} ${theme.spacing.mobile.sm}`;
      case 'medium': return `${theme.spacing.mobile.sm} ${theme.spacing.mobile.md}`;
      case 'large': return `${theme.spacing.mobile.md} ${theme.spacing.mobile.lg}`;
      default: return `${theme.spacing.mobile.sm} ${theme.spacing.mobile.md}`;
    }
  }};
  border-radius: ${({ $size }) => {
    switch ($size) {
      case 'small': return '12px';
      case 'medium': return '16px';
      case 'large': return '20px';
      default: return '16px';
    }
  }};
  font-size: ${({ theme, $size }) => {
    switch ($size) {
      case 'small': return theme.typography.fontSize.mobile.xs;
      case 'medium': return theme.typography.fontSize.mobile.sm;
      case 'large': return theme.typography.fontSize.mobile.md;
      default: return theme.typography.fontSize.mobile.sm;
    }
  }};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  transition: all 0.3s ease;
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
    animation: energyShimmer 3s infinite;
  }

  @keyframes energyShimmer {
    0% { left: -100%; }
    50% { left: 100%; }
    100% { left: 100%; }
  }
`;

const EnergyIcon = styled.span<{ $level: 'LOW' | 'MEDIUM' | 'HIGH'; $size: 'small' | 'medium' | 'large' }>`
  font-size: ${({ theme, $size }) => {
    switch ($size) {
      case 'small': return theme.typography.fontSize.mobile.sm;
      case 'medium': return theme.typography.fontSize.mobile.md;
      case 'large': return theme.typography.fontSize.mobile.lg;
      default: return theme.typography.fontSize.mobile.md;
    }
  }};
  animation: ${({ $level }) => {
    switch ($level) {
      case 'HIGH': return 'energyPulse 1.5s ease-in-out infinite';
      case 'MEDIUM': return 'energyPulse 2.5s ease-in-out infinite';
      case 'LOW': return 'none';
      default: return 'none';
    }
  }};

  @keyframes energyPulse {
    0%, 100% {
      transform: scale(1);
      filter: brightness(1);
    }
    50% {
      transform: scale(1.2);
      filter: brightness(1.3);
    }
  }
`;

const EnergyText = styled.span<{ $level: 'LOW' | 'MEDIUM' | 'HIGH' }>`
  color: ${({ $level }) => {
    switch ($level) {
      case 'HIGH': return '#7FA870';
      case 'MEDIUM': return '#F4A261';
      case 'LOW': return '#C85A5A';
      default: return '#7FA870';
    }
  }};
`;

const EnergyPercentage = styled.span`
  opacity: 0.8;
  font-weight: ${({ theme }) => theme.typography.fontWeight.normal};
`;

const StyledEnergyContainer = styled(EnergyContainer)<{ $level: 'LOW' | 'MEDIUM' | 'HIGH' }>`
  background: ${({ $level }) => {
    switch ($level) {
      case 'HIGH': return 'rgba(127, 168, 112, 0.1)';
      case 'MEDIUM': return 'rgba(244, 162, 97, 0.1)';
      case 'LOW': return 'rgba(200, 90, 90, 0.1)';
      default: return 'rgba(127, 168, 112, 0.1)';
    }
  }};
  color: ${({ $level }) => {
    switch ($level) {
      case 'HIGH': return '#7FA870';
      case 'MEDIUM': return '#F4A261';
      case 'LOW': return '#C85A5A';
      default: return '#7FA870';
    }
  }};
  border: 1px solid ${({ $level }) => {
    switch ($level) {
      case 'HIGH': return 'rgba(127, 168, 112, 0.2)';
      case 'MEDIUM': return 'rgba(244, 162, 97, 0.2)';
      case 'LOW': return 'rgba(200, 90, 90, 0.2)';
      default: return 'rgba(127, 168, 112, 0.2)';
    }
  }};

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
`;

const getEnergyIcon = (level: 'LOW' | 'MEDIUM' | 'HIGH') => {
  switch (level) {
    case 'HIGH': return '⚡';
    case 'MEDIUM': return '🔋';
    case 'LOW': return '🪫';
    default: return '⚡';
  }
};

const getEnergyText = (level: 'LOW' | 'MEDIUM' | 'HIGH') => {
  switch (level) {
    case 'HIGH': return 'High Energy';
    case 'MEDIUM': return 'Medium Energy';
    case 'LOW': return 'Low Energy';
    default: return 'High Energy';
  }
};

const getEnergyColor = (level: 'LOW' | 'MEDIUM' | 'HIGH') => {
  switch (level) {
    case 'HIGH': return '#7FA870';
    case 'MEDIUM': return '#F4A261';
    case 'LOW': return '#C85A5A';
    default: return '#7FA870';
  }
};

export const EnergyIndicator: React.FC<EnergyIndicatorProps> = ({
  energyLevel,
  score,
  showPercentage = false,
  size = 'medium',
  className,
}) => {
  const displayScore = score || 0;
  const icon = getEnergyIcon(energyLevel);
  const text = getEnergyText(energyLevel);
  const color = getEnergyColor(energyLevel);

  return (
    <StyledEnergyContainer
      $level={energyLevel}
      $size={size}
      className={className}
      title={`${text} Energy Level - ${displayScore}%`}
    >
      <EnergyIcon $level={energyLevel} $size={size}>
        {icon}
      </EnergyIcon>
      <EnergyText $level={energyLevel}>
        {text}
        {showPercentage && (
          <EnergyPercentage> ({displayScore}%)</EnergyPercentage>
        )}
      </EnergyText>
    </StyledEnergyContainer>
  );
};

export type { EnergyIndicatorProps };