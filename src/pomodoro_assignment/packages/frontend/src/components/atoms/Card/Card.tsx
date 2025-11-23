import React from 'react';
import styled from 'styled-components';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  hover?: boolean;
  padding?: 'none' | 'small' | 'medium' | 'large';
  elevation?: 'none' | 'small' | 'medium' | 'large';
  borderRadius?: 'none' | 'small' | 'medium' | 'large';
  backgroundColor?: string;
  borderColor?: string;
  fullWidth?: boolean;
  style?: React.CSSProperties;
}

const StyledCard = styled.div<{
  $hover?: boolean;
  $padding: 'none' | 'small' | 'medium' | 'large';
  $elevation: 'none' | 'small' | 'medium' | 'large';
  $borderRadius: 'none' | 'small' | 'medium' | 'large';
  $backgroundColor?: string;
  $borderColor?: string;
  $fullWidth?: boolean;
  $clickable?: boolean;
}>`
  ${({ theme, $backgroundColor, $borderColor, $padding, $elevation, $borderRadius, $fullWidth, $clickable }) => ({
    ...theme.components.Card,
    backgroundColor: $backgroundColor || theme.components.Card.backgroundColor,
    borderRadius: (() => {
      switch ($borderRadius) {
        case 'none': return '0';
        case 'small': return theme.borderRadius.sm;
        case 'large': return theme.borderRadius.lg;
        default: return theme.borderRadius.md;
      }
    })(),
    padding: (() => {
      switch ($padding) {
        case 'none': return '0';
        case 'small': return theme.spacing.sm;
        case 'large': return theme.spacing.xl;
        default: return theme.spacing.lg;
      }
    })(),
    boxShadow: (() => {
      switch ($elevation) {
        case 'none': return 'none';
        case 'small': return theme.shadows.sm;
        case 'large': return theme.shadows.lg;
        default: return theme.shadows.md;
      }
    })(),
    width: $fullWidth ? '100%' : 'auto',
    cursor: $clickable ? 'pointer' : 'default',
    border: $borderColor ? `1px solid ${$borderColor}` : 'none',
  })};

  ${({ $hover, $clickable, theme }) => ($hover || $clickable) && `
    &:hover {
      transform: translateY(-2px);
      box-shadow: ${theme.shadows.lg};
    }
  `}

  ${({ theme }) => theme.mediaQueries.mobile} {
    padding: ${({ theme, $padding }) => {
      switch ($padding) {
        case 'none': return '0';
        case 'small': return theme.spacing.sm;
        case 'large': return theme.spacing.lg;
        default: return theme.spacing.md;
      }
    }};
  }
`;

export const Card: React.FC<CardProps> = ({
  children,
  className,
  onClick,
  hover = false,
  padding = 'medium',
  elevation = 'medium',
  borderRadius = 'medium',
  backgroundColor,
  borderColor,
  fullWidth = false,
  style,
}) => {
  return (
    <StyledCard
      className={className}
      $hover={hover}
      $padding={padding}
      $elevation={elevation}
      $borderRadius={borderRadius}
      $backgroundColor={backgroundColor}
      $borderColor={borderColor}
      $fullWidth={fullWidth}
      $clickable={!!onClick}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
      style={style}
    >
      {children}
    </StyledCard>
  );
};

export type { CardProps };