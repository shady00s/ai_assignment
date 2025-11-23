import React, { useMemo } from 'react';

interface CircularTimerProps {
  size?: number;
  strokeWidth?: number;
  showControls?: boolean;
  progress?: number;
  remainingTime?: number;
  sessionType?: 'POMODORO' | 'SHORT_BREAK' | 'LONG_BREAK';
  className?: string;
}

export const CircularTimer: React.FC<CircularTimerProps> = ({
  size = 280,
  strokeWidth = 12,
  showControls = true,
  progress = 0,
  remainingTime = 1500,
  sessionType = 'POMODORO',
  className,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;

  // Calculate SVG stroke-dashoffset for smooth animation
  const svgProgress = useMemo(() => {
    return circumference * (1 - progress);
  }, [circumference, progress]);

  // Format time display with proper zero padding
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Get session type display text and colors based on zen design principles
  const getSessionTheme = () => {
    switch (sessionType) {
      case 'POMODORO':
        return {
          text: 'Focus Time',
          primaryColor: '#7FA870', // Sage green - calming, focused
          secondaryColor: '#8FBC8F', // Light sage
          bgGradient: 'linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%)',
          icon: '🍅',
        };
      case 'SHORT_BREAK':
        return {
          text: 'Short Break',
          primaryColor: '#F4A261', // Warm orange - gentle energy
          secondaryColor: '#F5B789',
          bgGradient: 'linear-gradient(135deg, #F4A261 0%, #F5B789 100%)',
          icon: '☕',
        };
      case 'LONG_BREAK':
        return {
          text: 'Long Break',
          primaryColor: '#E9C46A', // Soft yellow - restful
          secondaryColor: '#EED989',
          bgGradient: 'linear-gradient(135deg, #E9C46A 0%, #EED989 100%)',
          icon: '🌿',
        };
      default:
        return {
          text: 'Focus Time',
          primaryColor: '#7FA870',
          secondaryColor: '#8FBC8F',
          bgGradient: 'linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%)',
          icon: '🍅',
        };
    }
  };

  const sessionTheme = getSessionTheme();

  return (
    <div className={className} style={{ fontFamily: "'Inter', sans-serif" }}>
      <div style={{
        width: size,
        height: size,
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        {/* SVG Circular Progress */}
        <svg
          width={size}
          height={size}
          style={{
            transform: 'rotate(-90deg)',
            filter: 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1)',
          }}
        >
          {/* Background circle with zen-inspired neutral color */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            strokeWidth={strokeWidth}
            fill="none"
            stroke="#F0E6DC" // Soft cream background - zen aesthetic
            opacity={0.3}
          />

          {/* Progress circle with gradient and smooth animation */}
          <defs>
            <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={sessionTheme.primaryColor} />
              <stop offset="100%" stopColor={sessionTheme.secondaryColor} />
            </linearGradient>
          </defs>

          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            strokeWidth={strokeWidth}
            fill="none"
            stroke="url(#progressGradient)"
            strokeDasharray={circumference}
            strokeDashoffset={svgProgress}
            strokeLinecap="round"
            style={{
              transition: 'stroke-dashoffset 0.5s ease-in-out',
              transform: 'translateZ(0)', // Hardware acceleration for smooth animation
            }}
          />
        </svg>

        {/* Time and session type display */}
        <div style={{
          position: 'absolute',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 2,
        }}>
          {/* Time display */}
          <div style={{
            fontSize: Math.floor(size / 7),
            fontWeight: '700',
            color: '#2C3E50',
            fontFamily: 'Lora, serif',
            textAlign: 'center',
            lineHeight: 1.1,
            letterSpacing: '-0.02em',
          }}>
            {formatTime(remainingTime)}
          </div>

          {/* Session type */}
          <div style={{
            fontSize: Math.floor(size / 18),
            color: '#8B7D7B',
            marginTop: '6px',
            fontFamily: 'Inter, sans-serif',
            fontWeight: '500',
            textAlign: 'center',
          }}>
            {sessionTheme.text}
          </div>
        </div>

        {/* Zen element - subtle leaf icon */}
        <div style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: 0.08, // Very subtle - zen principle of minimalism
          fontSize: `${size * 0.45}px`,
          pointerEvents: 'none',
          zIndex: 1,
          filter: 'blur(0.5px)', // Soft, dreamy effect
        }}>
          {sessionTheme.icon}
        </div>

        {/* Subtle pulsing animation when timer is running */}
        <div style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          borderRadius: '50%',
          border: '1px solid rgba(127, 168, 112, 0.1)',
          animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          pointerEvents: 'none',
        }} />
      </div>

      {/* Enhanced Control Buttons */}
      {showControls && (
        <div style={{
          display: 'flex',
          gap: '12px',
          marginTop: '32px',
          justifyContent: 'center',
          flexWrap: 'wrap',
        }}>
          <button
            style={{
              padding: '14px 24px',
              background: sessionTheme.primaryColor,
              color: 'white',
              border: 'none',
              borderRadius: '16px',
              fontSize: '15px',
              fontWeight: '600',
              cursor: 'pointer',
              minWidth: '120px',
              transition: 'all 0.2s ease',
              boxShadow: '0 4px 12px rgba(127, 168, 112, 0.25)',
              fontFamily: 'Inter, sans-serif',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 6px 16px rgba(127, 168, 112, 0.35)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(127, 168, 112, 0.25)';
            }}
          >
            ▶️ Start
          </button>

          <button
            style={{
              padding: '14px 24px',
              background: '#F4A261',
              color: 'white',
              border: 'none',
              borderRadius: '16px',
              fontSize: '15px',
              fontWeight: '600',
              cursor: 'pointer',
              minWidth: '120px',
              transition: 'all 0.2s ease',
              boxShadow: '0 4px 12px rgba(244, 162, 97, 0.25)',
              fontFamily: 'Inter, sans-serif',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 6px 16px rgba(244, 162, 97, 0.35)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(244, 162, 97, 0.25)';
            }}
          >
            ⏸️ Pause
          </button>

          <button
            style={{
              padding: '14px 24px',
              background: 'transparent',
              color: '#8B7D7B',
              border: '2px solid #D4C4B0',
              borderRadius: '16px',
              fontSize: '15px',
              fontWeight: '600',
              cursor: 'pointer',
              minWidth: '120px',
              transition: 'all 0.2s ease',
              fontFamily: 'Inter, sans-serif',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = '#8B7D7B';
              e.currentTarget.style.background = 'rgba(139, 125, 123, 0.05)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = '#D4C4B0';
              e.currentTarget.style.background = 'transparent';
            }}
          >
            ⏭️ Skip
          </button>

          <button
            style={{
              padding: '14px 24px',
              background: '#C85A5A',
              color: 'white',
              border: 'none',
              borderRadius: '16px',
              fontSize: '15px',
              fontWeight: '600',
              cursor: 'pointer',
              minWidth: '120px',
              transition: 'all 0.2s ease',
              boxShadow: '0 4px 12px rgba(200, 90, 90, 0.25)',
              fontFamily: 'Inter, sans-serif',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 6px 16px rgba(200, 90, 90, 0.35)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(200, 90, 90, 0.25)';
            }}
          >
            ✅ Complete
          </button>
        </div>
      )}

      {/* Global styles for animation */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% {
              opacity: 0;
              transform: scale(0.95);
            }
            50% {
              opacity: 1;
              transform: scale(1.05);
            }
          }
        `}
      </style>
    </div>
  );
};

export type { CircularTimerProps };