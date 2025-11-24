import React, { useState } from 'react';
import styled, { keyframes } from 'styled-components';

interface WaterGlassProps {
  filled: boolean;
  onClick?: () => void;
  onRightClick?: (e: React.MouseEvent) => void;
  disabled?: boolean;
  animationDelay?: number;
  size?: 'small' | 'medium' | 'large';
}

const fillAnimation = keyframes`
  0% {
    transform: translateY(100%);
  }
  100% {
    transform: translateY(0);
  }
`;

const waveAnimation = keyframes`
  0%, 100% {
    transform: translateX(0) translateY(0);
  }
  25% {
    transform: translateX(-2px) translateY(-1px);
  }
  50% {
    transform: translateX(2px) translateY(-2px);
  }
  75% {
    transform: translateX(-1px) translateY(-1px);
  }
`;

const GlassContainer = styled.div<{ size?: 'small' | 'medium' | 'large' }>`
  position: relative;
  width: ${({ size }) => {
    switch (size) {
      case 'small': return '40px';
      case 'large': return '70px';
      default: return '60px';
    }
  }};
  height: ${({ size }) => {
    switch (size) {
      case 'small': return '50px';
      case 'large': return '90px';
      default: return '75px';
    }
  }};
  cursor: ${({ onClick, disabled }) => (onClick && !disabled) ? 'pointer' : 'default'};
  opacity: ${({ disabled }) => disabled ? 0.5 : 1};
  transition: transform 0.2s ease, opacity 0.2s ease;

  &:hover {
    transform: ${({ onClick, disabled }) => (onClick && !disabled) ? 'scale(1.05)' : 'scale(1)'};
  }

  &:active {
    transform: ${({ onClick, disabled }) => (onClick && !disabled) ? 'scale(0.95)' : 'scale(1)'};
  }
`;

const GlassShape = styled.div<{ size?: 'small' | 'medium' | 'large' }>`
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    rgba(255, 255, 255, 0.1) 0%,
    rgba(255, 255, 255, 0.05) 50%,
    rgba(255, 255, 255, 0) 100%
  );
  border: 2px solid #e0e0e0;
  border-radius: 0 0 8px 8px;
  box-shadow:
    inset 0 -2px 4px rgba(0, 0, 0, 0.1),
    0 2px 4px rgba(0, 0, 0, 0.1);

  &::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    height: 8px;
    background: #e0e0e0;
    border-radius: 4px 4px 0 0;
  }
`;

const WaterFill = styled.div<{
  filled: boolean;
  animationDelay?: number;
  size?: 'small' | 'medium' | 'large';
}>`
  position: absolute;
  bottom: 0;
  left: 2px;
  right: 2px;
  height: ${({ filled }) => (filled ? '85%' : '0%')};
  background: linear-gradient(
    to bottom,
    rgba(107, 142, 159, 0.8) 0%,
    rgba(107, 142, 159, 1) 100%
  );
  border-radius: 0 0 6px 6px;
  animation: ${({ filled, animationDelay }) =>
    filled ? `${fillAnimation} 0.6s ease-out ${animationDelay}ms both` : 'none'};
  transition: height 0.3s ease-out;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 8px;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    transform: translateX(-50%) scale(0);
    left: 50%;
    animation: ${({ filled }) => filled ? `${waveAnimation} 2s ease-in-out infinite` : 'none'};
  }
`;

const WaterHighlight = styled.div<{ filled: boolean; size?: 'small' | 'medium' | 'large' }>`
  position: absolute;
  top: ${({ filled, size }) => filled ? (size === 'small' ? '15%' : '20%') : '50%'};
  left: 20%;
  width: 30%;
  height: 40%;
  background: linear-gradient(
    135deg,
    rgba(255, 255, 255, 0.6) 0%,
    rgba(255, 255, 255, 0.2) 50%,
    rgba(255, 255, 255, 0) 100%
  );
  border-radius: 50%;
  transform: rotate(-15deg);
  opacity: ${({ filled }) => (filled ? 1 : 0)};
  transition: opacity 0.3s ease, top 0.3s ease;
`;

const Bubbles = styled.div<{ filled: boolean }>`
  position: absolute;
  bottom: 10%;
  left: 0;
  right: 0;
  display: ${({ filled }) => (filled ? 'block' : 'none')};

  &::before,
  &::after {
    content: '';
    position: absolute;
    background: rgba(255, 255, 255, 0.4);
    border-radius: 50%;
    animation: ${({ filled }) => filled ? `${waveAnimation} 1.5s ease-in-out infinite` : 'none'};
  }

  &::before {
    width: 4px;
    height: 4px;
    left: 25%;
    animation-delay: 0.2s;
  }

  &::after {
    width: 3px;
    height: 3px;
    left: 70%;
    animation-delay: 0.7s;
  }
`;

const EmptyGlassIndicator = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 20px;
  opacity: 0.3;
  user-select: none;
`;

export const WaterGlass: React.FC<WaterGlassProps> = ({
  filled,
  onClick,
  onRightClick,
  disabled = false,
  animationDelay = 0,
  size = 'medium',
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    if (onClick && !disabled) {
      onClick();
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    if (onRightClick && !disabled) {
      onRightClick(e);
    }
  };

  return (
    <GlassContainer
      size={size}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
      disabled={disabled}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      role="button"
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={(e) => {
        if ((e.key === 'Enter' || e.key === ' ') && onClick && !disabled) {
          e.preventDefault();
          onClick();
        }
      }}
      aria-label={`${filled ? 'Filled' : 'Empty'} water glass ${onClick ? '- click to fill' : ''}`}
      aria-pressed={filled}
    >
      <GlassShape size={size}>
        <WaterFill
          filled={filled}
          animationDelay={animationDelay}
          size={size}
        />
        <WaterHighlight filled={filled} size={size} />
        <Bubbles filled={filled} />
      </GlassShape>

      {!filled && (
        <EmptyGlassIndicator>
          💧
        </EmptyGlassIndicator>
      )}

      {/* Hover tooltip for clickable glasses */}
      {isHovered && onClick && !filled && !disabled && (
        <div
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%) translateY(-4px)',
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            whiteSpace: 'nowrap',
            zIndex: 1000,
          }}
        >
          Click to fill
        </div>
      )}

      {/* Hover tooltip for right-clickable glasses */}
      {isHovered && onRightClick && filled && !disabled && (
        <div
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%) translateY(-4px)',
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            whiteSpace: 'nowrap',
            zIndex: 1000,
          }}
        >
          Right-click to empty
        </div>
      )}
    </GlassContainer>
  );
};