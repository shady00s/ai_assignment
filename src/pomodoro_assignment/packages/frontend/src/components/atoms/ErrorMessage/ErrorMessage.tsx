import React from 'react';
import styled from 'styled-components';

const ErrorContainer = styled.div<{ $variant?: 'default' | 'card' }>`
  padding: ${({ theme, $variant }) => $variant === 'card' ? theme.spacing.lg : theme.spacing.md};
  background-color: ${({ theme, $variant }) => $variant === 'card' ? '#FFFFFF' : theme.colors.error.light};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border: ${({ theme, $variant }) => $variant === 'card' ? `1px solid ${theme.colors.error.main}` : 'none'};
  margin: ${({ theme }) => theme.spacing.md} 0;
`;

const ErrorHeader = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const ErrorIcon = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  margin-right: ${({ theme }) => theme.spacing.sm};
  color: ${({ theme }) => theme.colors.error.main};
`;

const ErrorTitle = styled.h3`
  color: ${({ theme }) => theme.colors.error.main};
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  margin: 0;
`;

const MessageText = styled.p<{ $variant?: 'default' | 'card' }>`
  color: ${({ theme, $variant }) => $variant === 'card' ? theme.colors.neutral[500] : theme.colors.error.dark};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  line-height: 1.5;
  margin: 0;
`;

const RetryButton = styled.button`
  margin-top: ${({ theme }) => theme.spacing.md};
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  background-color: ${({ theme }) => theme.colors.error.main};
  color: white;
  border: none;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  cursor: pointer;
  transition: background-color 0.2s ease;

  &:hover {
    background-color: ${({ theme }) => theme.colors.error.dark};
  }
`;

export interface ErrorMessageProps {
  message: string;
  title?: string;
  variant?: 'default' | 'card';
  onRetry?: () => void;
  retryText?: string;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  message,
  title = 'Oops! Something went wrong',
  variant = 'default',
  onRetry,
  retryText = 'Try Again'
}) => {
  return (
    <ErrorContainer $variant={variant}>
      <ErrorHeader>
        <ErrorIcon>⚠️</ErrorIcon>
        <ErrorTitle>{title}</ErrorTitle>
      </ErrorHeader>
      <MessageText $variant={variant}>{message}</MessageText>
      {onRetry && (
        <RetryButton onClick={onRetry}>{retryText}</RetryButton>
      )}
    </ErrorContainer>
  );
};