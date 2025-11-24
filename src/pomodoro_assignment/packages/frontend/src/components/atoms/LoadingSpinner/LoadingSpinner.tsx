import React from 'react';
import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const LoadingContainer = styled.div<{ $centered?: boolean }>`
  display: flex;
  justify-content: ${({ $centered }) => $centered ? 'center' : 'flex-start'};
  align-items: center;
  padding: ${({ theme, $centered }) => $centered ? theme.spacing.xl : '0'};
`;

const SpinnerIcon = styled.div<{ size: 'small' | 'medium' | 'large' }>`
  border: 3px solid ${({ theme }) => theme.colors.neutral[200]};
  border-top: 3px solid ${({ theme }) => theme.colors.primary.main};
  border-radius: 50%;
  width: ${({ size }) => {
    switch (size) {
      case 'small': return '24px';
      case 'medium': return '36px';
      case 'large': return '48px';
      default: return '36px';
    }
  }};
  height: ${({ size }) => {
    switch (size) {
      case 'small': return '24px';
      case 'medium': return '36px';
      case 'large': return '48px';
      default: return '36px';
    }
  }};
  animation: ${spin} 1s linear infinite;
`;

const LoadingText = styled.span<{ size: 'small' | 'medium' | 'large' }>`
  margin-left: ${({ theme }) => theme.spacing.sm};
  color: ${({ theme }) => theme.colors.neutral[400]};
  font-size: ${({ theme, size }) => {
    switch (size) {
      case 'small': return theme.typography.fontSize.sm;
      case 'medium': return theme.typography.fontSize.base;
      case 'large': return theme.typography.fontSize.lg;
      default: return theme.typography.fontSize.base;
    }
  }};
`;

export interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  centered?: boolean;
  text?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'medium',
  centered = false,
  text
}) => {
  return (
    <LoadingContainer $centered={centered}>
      <SpinnerIcon size={size} />
      {text && <LoadingText size={size}>{text}</LoadingText>}
    </LoadingContainer>
  );
};