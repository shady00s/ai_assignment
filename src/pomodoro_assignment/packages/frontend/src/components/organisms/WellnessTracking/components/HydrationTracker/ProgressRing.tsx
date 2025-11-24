import React from 'react';
import styled, { keyframes } from 'styled-components';

interface ProgressRingProps {
  percentage: number;
  size: number;
  strokeWidth: number;
  color: string;
  children?: React.ReactNode;
  isComplete?: boolean;
  showBackground?: boolean;
}

const pulseAnimation = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.05);
    opacity: 0.8;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

const RingContainer = styled.div<{ size: number; isComplete?: boolean }>`
  position: relative;
  width: ${({ size }) => size}px;
  height: ${({ size }) => size}px;
  animation: ${({ isComplete }) => isComplete ? `${pulseAnimation} 2s ease-in-out infinite` : 'none'};
`;

const RingSvg = styled.svg<{ size: number }>`
  position: absolute;
  top: 0;
  left: 0;
  width: ${({ size }) => size}px;
  height: ${({ size }) => size}px;
  transform: rotate(-90deg);
`;

const BackgroundCircle = styled.circle<{ strokeWidth: number; showBackground?: boolean }>`
  fill: none;
  stroke: ${({ showBackground, theme }) => (showBackground ? theme.colors.neutral[200] : 'transparent')};
  stroke-width: ${({ strokeWidth }) => strokeWidth}px;
  transition: stroke 0.3s ease;
`;

const ProgressCircle = styled.circle<{ strokeWidth: number; color: string; percentage: number }>`
  fill: none;
  stroke: ${({ color }) => color};
  stroke-width: ${({ strokeWidth }) => strokeWidth}px;
  stroke-linecap: round;
  stroke-dasharray: ${({ strokeWidth }) => {
    const radius = (100 - strokeWidth / 2) / 2;
    return 2 * Math.PI * radius;
  }}px;
  stroke-dashoffset: ${({ strokeWidth, percentage }) => {
    const radius = (100 - strokeWidth / 2) / 2;
    const circumference = 2 * Math.PI * radius;
    return circumference - (percentage / 100) * circumference;
  }}px;
  transition: stroke-dashoffset 0.5s ease-in-out;
`;

const ProgressContent = styled.div<{ size: number }>`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: ${({ size }) => size * 0.7}px;
  height: ${({ size }) => size * 0.7}px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  text-align: center;
`;

const CheckMark = styled.svg<{ size: number }>`
  width: ${({ size }) => size * 0.3}px;
  height: ${({ size }) => size * 0.3}px;
  animation: fadeInScale 0.3s ease-out;
`;

const fadeInScale = keyframes`
  from {
    opacity: 0;
    transform: scale(0.5);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
`;

export const ProgressRing: React.FC<ProgressRingProps> = ({
  percentage,
  size,
  strokeWidth,
  color,
  children,
  isComplete = false,
  showBackground = true,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <RingContainer size={size} isComplete={isComplete}>
      <RingSvg size={size} viewBox={`0 0 ${size} ${size}`}>
        <BackgroundCircle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          showBackground={showBackground}
        />
        <ProgressCircle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          color={color}
          percentage={percentage}
        />
      </RingSvg>

      <ProgressContent size={size}>
        {isComplete ? (
          <CheckMark size={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle
              cx="12"
              cy="12"
              r="10"
              stroke={color}
              strokeWidth="2"
              fill="rgba(255, 255, 255, 0.9)"
            />
            <path
              d="M9 12l2 2 4-4"
              stroke={color}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </CheckMark>
        ) : (
          children
        )}
      </ProgressContent>
    </RingContainer>
  );
};