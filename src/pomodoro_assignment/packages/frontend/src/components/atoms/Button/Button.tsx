import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  children: React.ReactNode;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
  className?: string;
  fullWidth?: boolean;
  style?: React.CSSProperties;
  onMouseEnter?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  onMouseLeave?: (e: React.MouseEvent<HTMLButtonElement>) => void;
}

const StyledButton = styled(motion.button)<{
  $variant: 'primary' | 'secondary' | 'ghost';
  $size: 'small' | 'medium' | 'large';
  $fullWidth?: boolean;
}>`
  ${({ theme, $variant, $size, $fullWidth }) => theme.components.Button[$variant]};
  font-size: ${({ theme, $size }) => {
    switch ($size) {
      case 'small': return theme.typography.fontSize.sm;
      case 'medium': return theme.typography.fontSize.base;
      case 'large': return theme.typography.fontSize.lg;
      default: return theme.typography.fontSize.base;
    }
  }};
  padding: ${({ theme, $size }) => {
    switch ($size) {
      case 'small': return `${theme.spacing.xs} ${theme.spacing.sm}`;
      case 'medium': return `${theme.spacing.sm} ${theme.spacing.md}`;
      case 'large': return `${theme.spacing.md} ${theme.spacing.lg}`;
      default: return `${theme.spacing.sm} ${theme.spacing.md}`;
    }
  }};
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  text-decoration: none;
  position: relative;
  overflow: hidden;
  width: ${({ $fullWidth }) => $fullWidth ? '100%' : 'auto'};
  min-height: ${({ $size }) => {
    switch ($size) {
      case 'small': return '32px';
      case 'medium': return '40px';
      case 'large': return '48px';
      default: return '40px';
    }
  }};

  &:focus {
    outline: 2px solid ${({ theme }) => theme.colors.primary.main};
    outline-offset: 2px;
  }

  &:disabled {
    cursor: not-allowed;
    opacity: 0.6;
  }

  ${({ theme }) => theme.mediaQueries.mobile} {
    min-height: 44px; /* Touch-friendly minimum size */
    font-size: ${({ theme }) => theme.typography.fontSize.base};
  }
`;

const LoadingSpinner = styled.div`
  width: 16px;
  height: 16px;
  border: 2px solid transparent;
  border-top: 2px solid currentColor;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

export type { ButtonProps };

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  icon,
  children,
  onClick,
  type = 'button',
  className,
  fullWidth = false,
  style,
  onMouseEnter,
  onMouseLeave,
}) => {
  return (
    <StyledButton
      as={motion.button}
      $variant={variant}
      $size={size}
      $fullWidth={fullWidth}
      disabled={disabled || loading}
      onClick={onClick}
      type={type}
      className={className}
      style={style}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      whileHover={{ scale: loading ? 1 : 1.02 }}
      whileTap={{ scale: loading ? 1 : 0.98 }}
      transition={{ duration: 0.15 }}
    >
      {loading && <LoadingSpinner />}
      {!loading && icon}
      {children}
    </StyledButton>
  );
};