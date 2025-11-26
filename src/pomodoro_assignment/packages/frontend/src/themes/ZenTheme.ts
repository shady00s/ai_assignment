import { DefaultTheme } from 'styled-components';
import { designTokens } from './designTokens';

export const ZenTheme: DefaultTheme = {
  ...designTokens,
  // Use the comprehensive responsive breakpoints from designTokens
  breakpoints: designTokens.breakpoints,
  mediaQueries: designTokens.mediaQueries,
  components: {
    Button: {
      primary: {
        background: designTokens.colors.accent.gradient.ocean,
        color: '#FFFFFF',
        borderRadius: designTokens.borderRadius.lg,
        padding: `${designTokens.spacing.md} ${designTokens.spacing.xl}`,
        border: 'none',
        cursor: 'pointer',
        fontFamily: designTokens.typography.fontFamily.body,
        fontWeight: designTokens.typography.fontWeight.semibold,
        fontSize: designTokens.typography.fontSize.body,
        transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.expo}`,
        boxShadow: designTokens.shadows.primary,
        position: 'relative',
        overflow: 'hidden',

        '&::before': {
          content: '""',
          position: 'absolute',
          top: '0',
          left: '-100%',
          width: '100%',
          height: '100%',
          background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
          transition: 'left 0.5s',
        },

        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: designTokens.shadows.xl,

          '&::before': {
            left: '100%',
          },
        },

        '&:active': {
          transform: 'translateY(0)',
        },

        '&:disabled': {
          opacity: 0.6,
          cursor: 'not-allowed',
          transform: 'none',
          boxShadow: designTokens.shadows.md,
        },
      },
      secondary: {
        background: designTokens.colors.glass.background,
        backdropFilter: `blur(${designTokens.colors.glass.blur})`,
        color: designTokens.colors.primary[600],
        border: `1px solid ${designTokens.colors.glass.border}`,
        borderRadius: designTokens.borderRadius.lg,
        padding: `${designTokens.spacing.md} ${designTokens.spacing.xl}`,
        cursor: 'pointer',
        fontFamily: designTokens.typography.fontFamily.body,
        fontWeight: designTokens.typography.fontWeight.semibold,
        fontSize: designTokens.typography.fontSize.body,
        transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.expo}`,

        '&:hover': {
          background: designTokens.colors.primary[50],
          borderColor: designTokens.colors.primary[200],
          transform: 'translateY(-1px)',
          boxShadow: designTokens.shadows.lg,
        },
      },
      ghost: {
        background: 'transparent',
        color: designTokens.colors.neutral[600],
        border: 'none',
        borderRadius: designTokens.borderRadius.md,
        padding: `${designTokens.spacing.sm} ${designTokens.spacing.md}`,
        cursor: 'pointer',
        fontFamily: designTokens.typography.fontFamily.body,
        fontWeight: designTokens.typography.fontWeight.medium,
        fontSize: designTokens.typography.fontSize.body,
        transition: `all ${designTokens.animation.duration.fast} ${designTokens.animation.easing.smooth}`,

        '&:hover': {
          background: designTokens.colors.neutral[100],
          color: designTokens.colors.neutral[900],
        },
      },
      // Modern accent button with neon glow
      accent: {
        background: designTokens.colors.accent.gradient.sunset,
        color: '#FFFFFF',
        borderRadius: designTokens.borderRadius.full,
        padding: `${designTokens.spacing.md} ${designTokens.spacing.xl}`,
        border: 'none',
        cursor: 'pointer',
        fontFamily: designTokens.typography.fontFamily.body,
        fontWeight: designTokens.typography.fontWeight.bold,
        fontSize: designTokens.typography.fontSize.body,
        transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.bouncy}`,
        boxShadow: designTokens.shadows.glow.warning,

        '&:hover': {
          transform: 'scale(1.05) translateY(-2px)',
          boxShadow: designTokens.shadows.glow.error,
        },
      },
    },
    Card: {
      background: designTokens.colors.glass.background,
      backdropFilter: `blur(${designTokens.colors.glass.blur})`,
      border: `1px solid ${designTokens.colors.glass.border}`,
      borderRadius: designTokens.borderRadius.xl,
      boxShadow: designTokens.shadows.lg,
      padding: designTokens.spacing.lg,
      transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.expo}`,

      '&:hover': {
        transform: 'translateY(-4px)',
        boxShadow: designTokens.shadows.xl,
        borderColor: designTokens.colors.primary[200],
      },
    },
    Timer: {
      fontSize: designTokens.typography.fontSize.mobile.h2, // Use mobile size for mobile-first
      fontWeight: designTokens.typography.fontWeight.bold,
      color: designTokens.colors.neutral[800],
      fontFamily: designTokens.typography.fontFamily.heading,
      textAlign: 'center',
      textShadow: designTokens.shadows.sm,

      // Responsive timer
      [designTokens.mediaQueries.tablet]: {
        fontSize: designTokens.typography.fontSize.tablet.h2,
      },

      [designTokens.mediaQueries.desktop]: {
        fontSize: designTokens.typography.fontSize.desktop.h2,
      },
    },
    Input: {
      background: designTokens.colors.glass.background,
      backdropFilter: `blur(${designTokens.colors.glass.blur})`,
      border: `1px solid ${designTokens.colors.glass.border}`,
      borderRadius: designTokens.borderRadius.lg,
      padding: `${designTokens.spacing.md} ${designTokens.spacing.lg}`,
      fontSize: designTokens.typography.fontSize.body,
      color: designTokens.colors.neutral[800],
      fontFamily: designTokens.typography.fontFamily.body,
      transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.smooth}`,

      '&:focus': {
        outline: 'none',
        borderColor: designTokens.colors.primary[400],
        boxShadow: `0 0 0 3px ${designTokens.colors.primary[100]}`,
        transform: 'translateY(-1px)',
      },

      '&::placeholder': {
        color: designTokens.colors.neutral[400],
      },
    },
    Navigation: {
      background: designTokens.colors.glass.background,
      backdropFilter: `blur(${designTokens.colors.glass.blur})`,
      borderBottom: `1px solid ${designTokens.colors.glass.border}`,
      boxShadow: designTokens.shadows.sm,
      padding: `${designTokens.spacing.mobile.sm} 0`,
      position: 'sticky',
      top: '0',
      zIndex: 100,

      // Responsive navigation
      [designTokens.mediaQueries.mobile]: {
        padding: `${designTokens.spacing.mobile.xs} 0`,
      },

      [designTokens.mediaQueries.tablet]: {
        padding: `${designTokens.spacing.tablet.sm} 0`,
      },

      [designTokens.mediaQueries.desktop]: {
        padding: `${designTokens.spacing.sm} 0`,
      },
    },
    ZenGarden: {
      background: `linear-gradient(135deg, ${designTokens.colors.neutral[50]} 0%, ${designTokens.colors.primary[50]} 100%)`,
      borderRadius: designTokens.borderRadius.xl,
      padding: designTokens.spacing.xl,
      minHeight: '200px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      position: 'relative',
      overflow: 'hidden',
      boxShadow: designTokens.shadows.lg,

      '&::before': {
        content: '""',
        position: 'absolute',
        top: '-50%',
        left: '-50%',
        width: '200%',
        height: '200%',
        background: `radial-gradient(circle, ${designTokens.colors.primary[200]}20 0%, transparent 70%)`,
        animation: designTokens.animation.presets.float,
      },
    },
    ProgressRing: {
      transform: 'rotate(-90deg)',
      filter: 'drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1))',
    },
    TaskCard: {
      background: designTokens.colors.glass.background,
      backdropFilter: `blur(${designTokens.colors.glass.blur})`,
      border: `1px solid ${designTokens.colors.glass.border}`,
      borderLeft: `4px solid ${designTokens.colors.primary[500]}`,
      borderRadius: designTokens.borderRadius.lg,
      boxShadow: designTokens.shadows.md,
      padding: designTokens.spacing.mobile.md,
      marginBottom: designTokens.spacing.mobile.sm,
      transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.expo}`,

      '&:hover': {
        transform: 'translateX(4px) translateY(-2px)',
        boxShadow: designTokens.shadows.lg,
        borderLeftColor: designTokens.colors.primary[600],
      },

      // Responsive task cards
      [designTokens.mediaQueries.tablet]: {
        padding: designTokens.spacing.tablet.md,
        marginBottom: designTokens.spacing.tablet.sm,
      },

      [designTokens.mediaQueries.desktop]: {
        padding: designTokens.spacing.md,
        marginBottom: designTokens.spacing.sm,
      },

      '&.high-priority': {
        borderLeftColor: designTokens.colors.error[500],
        '&:hover': {
          borderLeftColor: designTokens.colors.error[600],
        },
      },

      '&.medium-priority': {
        borderLeftColor: designTokens.colors.warning[500],
        '&:hover': {
          borderLeftColor: designTokens.colors.warning[600],
        },
      },

      '&.low-priority': {
        borderLeftColor: designTokens.colors.success[500],
        '&:hover': {
          borderLeftColor: designTokens.colors.success[600],
        },
      },

      '&.completed': {
        borderLeftColor: designTokens.colors.success[500],
        opacity: 0.8,
        background: designTokens.colors.success[50],
      },
    },
    AchievementBadge: {
      background: designTokens.colors.accent.gradient.aurora,
      color: '#FFFFFF',
      borderRadius: designTokens.borderRadius.full,
      padding: `${designTokens.spacing.sm} ${designTokens.spacing.lg}`,
      fontSize: designTokens.typography.fontSize.sm,
      fontWeight: designTokens.typography.fontWeight.semibold,
      fontFamily: designTokens.typography.fontFamily.body,
      display: 'inline-flex',
      alignItems: 'center',
      gap: designTokens.spacing.xs,
      boxShadow: designTokens.shadows.glow.success,
      transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.bouncy}`,

      '&:hover': {
        transform: 'scale(1.05)',
        boxShadow: designTokens.shadows.glow.primary,
      },
    },
    ProgressBar: {
      background: designTokens.colors.neutral[200],
      borderRadius: designTokens.borderRadius.full,
      height: '8px',
      overflow: 'hidden',
      position: 'relative',
      boxShadow: designTokens.shadows.inner,

      '&::after': {
        content: '""',
        position: 'absolute',
        top: '0',
        left: '0',
        height: '100%',
        background: designTokens.colors.accent.gradient.ocean,
        borderRadius: designTokens.borderRadius.full,
        transition: `width ${designTokens.animation.duration.normal} ${designTokens.animation.easing.expo}`,
        boxShadow: designTokens.shadows.glow.primary,
      },
    },
    // Modern glass surface for panels
    GlassSurface: {
      background: designTokens.colors.glass.background,
      backdropFilter: `blur(${designTokens.colors.glass.blur})`,
      border: `1px solid ${designTokens.colors.glass.border}`,
      borderRadius: designTokens.borderRadius.xl,
      boxShadow: designTokens.shadows.lg,
      transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.smooth}`,

      '&:hover': {
        borderColor: designTokens.colors.primary[200],
        boxShadow: designTokens.shadows.xl,
      },
    },
    // Hero section styling
    Hero: {
      background: designTokens.colors.accent.gradient.ocean,
      color: '#FFFFFF',
      padding: `${designTokens.spacing.xxl} ${designTokens.spacing.lg}`,
      borderRadius: designTokens.borderRadius.xl,
      textAlign: 'center',
      position: 'relative',
      overflow: 'hidden',
      boxShadow: designTokens.shadows.xl,

      '&::before': {
        content: '""',
        position: 'absolute',
        top: '0',
        left: '0',
        right: '0',
        bottom: '0',
        background: `radial-gradient(circle at 20% 80%, ${designTokens.colors.accent.neon.blue}20 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, ${designTokens.colors.accent.neon.purple}20 0%, transparent 50%)`,
        animation: designTokens.animation.presets.float,
      },

      '& > *': {
        position: 'relative',
        zIndex: 1,
      },
    },
  },
};

// Global theme extensions
export const darkTheme = {
  ...ZenTheme,
  colors: {
    ...ZenTheme.colors,
    background: ZenTheme.colors.dark.background,
    surface: ZenTheme.colors.dark.surface,
    card: ZenTheme.colors.dark.card,
    border: ZenTheme.colors.dark.border,
    text: ZenTheme.colors.dark.text,
    textSecondary: ZenTheme.colors.dark.textSecondary,
    primary: ZenTheme.colors.dark.accent,
  },
};

export const lightTheme = ZenTheme;

// Theme context type
export interface ThemeContextType {
  theme: DefaultTheme;
  isDark: boolean;
  toggleTheme: () => void;
}

// DefaultTheme will be automatically extended by the export below