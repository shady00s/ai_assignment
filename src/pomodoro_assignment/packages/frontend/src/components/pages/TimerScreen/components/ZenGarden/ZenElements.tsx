import React, { useEffect, useState } from 'react';
import styled from 'styled-components';

interface ZenElementsProps {
  isRunning: boolean;
  sessionType: 'POMODORO' | 'SHORT_BREAK' | 'LONG_BREAK';
  progress: number;
  className?: string;
}

interface Particle {
  id: number;
  x: number;
  y: number;
  size: number;
  opacity: number;
  speed: number;
  element: string;
}

const ZenElementsContainer = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  overflow: hidden;
  border-radius: inherit;
`;

const ParticleElement = styled.div<{ $x: number; $y: number; $size: number; $opacity: number; $element: string }>`
  position: absolute;
  left: ${({ $x }) => $x}%;
  top: ${({ $y }) => $y}%;
  font-size: ${({ $size }) => $size}px;
  opacity: ${({ $opacity }) => $opacity};
  animation: float ${({ $element }) => $element === '💧' ? '4s' : $element === '🍃' ? '6s' : '8s'} ease-in-out infinite;
  filter: blur(0.5px);
  z-index: 1;

  @keyframes float {
    0%, 100% {
      transform: translateY(0px) translateX(0px) rotate(0deg);
    }
    25% {
      transform: translateY(-10px) translateX(5px) rotate(90deg);
    }
    50% {
      transform: translateY(-5px) translateX(-5px) rotate(180deg);
    }
    75% {
      transform: translateY(-15px) translateX(3px) rotate(270deg);
    }
  }

  @keyframes ripple {
    0% {
      transform: scale(0.8);
      opacity: 0;
    }
    50% {
      opacity: 0.3;
    }
    100% {
      transform: scale(2);
      opacity: 0;
    }
  }

  @keyframes pulse {
    0%, 100% {
      transform: scale(1);
      opacity: 0.6;
    }
    50% {
      transform: scale(1.2);
      opacity: 0.3;
    }
  }

  @keyframes sway {
    0%, 100% {
      transform: translateX(0px) rotate(0deg);
    }
    25% {
      transform: translateX(-3px) rotate(-2deg);
    }
    75% {
      transform: translateX(3px) rotate(2deg);
    }
  }
`;

const SandPattern = styled.svg<{ $progress: number }>`
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 60px;
  opacity: 0.3;
  z-index: 0;

  .sand-line {
    stroke: #D4C4B0;
    stroke-width: 1;
    fill: none;
    stroke-dasharray: 300;
    stroke-dashoffset: ${({ $progress }) => 300 - (300 * $progress)};
    transition: stroke-dashoffset 0.5s ease-in-out;
  }
`;

const RippleEffect = styled.div<{ $active: boolean }>`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 100px;
  height: 100px;
  border: 2px solid rgba(127, 168, 112, 0.2);
  border-radius: 50%;
  opacity: ${({ $active }) => $active ? 1 : 0};
  animation: ${({ $active }) => $active ? 'ripple 2s ease-out infinite' : 'none'};
  z-index: 0;

  @keyframes ripple {
    0% {
      transform: translate(-50%, -50%) scale(0.5);
      opacity: 1;
    }
    100% {
      transform: translate(-50%, -50%) scale(3);
      opacity: 0;
    }
  }
`;

const ZenCircle = styled.div<{ $active: boolean; $progress: number }>`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: ${({ $progress }) => 100 + ($progress * 200)}px;
  height: ${({ $progress }) => 100 + ($progress * 200)}px;
  border: 1px solid rgba(127, 168, 112, ${({ $active, $progress }) => $active ? 0.3 - ($progress * 0.2) : 0.1});
  border-radius: 50%;
  opacity: ${({ $active }) => $active ? 0.8 : 0.2};
  z-index: 0;
  transition: all 0.3s ease;
`;

export const ZenElements: React.FC<ZenElementsProps> = ({
  isRunning,
  sessionType,
  progress,
  className,
}) => {
  const [particles, setParticles] = useState<Particle[]>([]);

  // Generate particles based on session type and progress
  useEffect(() => {
    if (!isRunning) {
      setParticles([]);
      return;
    }

    const elementTypes = {
      POMODORO: ['🌿', '🪴', '🍃'],
      SHORT_BREAK: ['💧', '☕', '💨'],
      LONG_BREAK: ['🌸', '🍂', '🌺'],
    };

    const availableElements = elementTypes[sessionType];
    const particleCount = Math.floor(progress * 8) + 2;

    const newParticles: Particle[] = Array.from({ length: particleCount }, (_, i) => ({
      id: Date.now() + i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: 12 + Math.random() * 8,
      opacity: 0.1 + Math.random() * 0.2,
      speed: 0.5 + Math.random() * 1.5,
      element: availableElements[Math.floor(Math.random() * availableElements.length)],
    }));

    setParticles(newParticles);
  }, [isRunning, sessionType, progress]);

  // Update particle positions
  useEffect(() => {
    if (!isRunning || particles.length === 0) return;

    const interval = setInterval(() => {
      setParticles(prevParticles =>
        prevParticles.map(particle => ({
          ...particle,
          y: particle.y <= -10 ? 110 : particle.y - particle.speed,
          x: particle.x + Math.sin(Date.now() / 1000 + particle.id) * 0.5,
        }))
      );
    }, 50);

    return () => clearInterval(interval);
  }, [isRunning, particles.length]);

  // Generate sand raking pattern
  const generateSandPath = () => {
    const paths = [];
    const lineCount = 5;

    for (let i = 0; i < lineCount; i++) {
      const y = 10 + (i * 12);
      const amplitude = 5 + Math.sin(progress * Math.PI) * 3;
      const frequency = 0.02 + (i * 0.005);

      let path = `M 0 ${y}`;
      for (let x = 0; x <= 100; x += 2) {
        const yOffset = Math.sin((x * frequency) + (progress * Math.PI * 2)) * amplitude;
        path += ` L ${x} ${y + yOffset}`;
      }

      paths.push(path);
    }

    return paths;
  };

  return (
    <ZenElementsContainer className={className}>
      {/* Sand raking pattern */}
      <SandPattern $progress={progress} viewBox="0 0 100 60" preserveAspectRatio="none">
        {generateSandPath().map((path, index) => (
          <path key={index} className="sand-line" d={path} />
        ))}
      </SandPattern>

      {/* Ripple effect for active sessions */}
      <RippleEffect $active={isRunning} />

      {/* Zen circle expansion */}
      <ZenCircle $active={isRunning} $progress={progress} />

      {/* Floating particles */}
      {particles.map((particle) => (
        <ParticleElement
          key={particle.id}
          $x={particle.x}
          $y={particle.y}
          $size={particle.size}
          $opacity={particle.opacity}
          $element={particle.element}
          style={{
            animation: particle.element === '💧'
              ? 'float 4s ease-in-out infinite'
              : particle.element === '🍃'
              ? 'sway 6s ease-in-out infinite'
              : 'pulse 8s ease-in-out infinite',
          }}
        >
          {particle.element}
        </ParticleElement>
      ))}
    </ZenElementsContainer>
  );
};

export type { ZenElementsProps };