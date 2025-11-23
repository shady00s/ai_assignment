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
        backgroundColor: designTokens.colors.primary.main,
        color: '#FFFFFF',
        borderRadius: designTokens.borderRadius.md,
        padding: `${designTokens.spacing.sm} ${designTokens.spacing.md}`,
        border: 'none',
        cursor: 'pointer',
        transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.easeInOut}`,

        '&:hover': {
          backgroundColor: designTokens.colors.primary.dark,
        },

        '&:disabled': {
          opacity: 0.6,
          cursor: 'not-allowed',
        },
      },
      secondary: {
        backgroundColor: 'transparent',
        color: designTokens.colors.primary.main,
        border: `2px solid ${designTokens.colors.primary.main}`,
        borderRadius: designTokens.borderRadius.md,
        padding: `${designTokens.spacing.sm} ${designTokens.spacing.md}`,
        cursor: 'pointer',
        transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.easeInOut}`,

        '&:hover': {
          backgroundColor: designTokens.colors.primary.main,
          color: '#FFFFFF',
        },
      },
      ghost: {
        backgroundColor: 'transparent',
        color: designTokens.colors.neutral[500],
        border: 'none',
        borderRadius: designTokens.borderRadius.md,
        padding: `${designTokens.spacing.sm} ${designTokens.spacing.md}`,
        cursor: 'pointer',
        transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.easeInOut}`,

        '&:hover': {
          backgroundColor: designTokens.colors.neutral[200],
        },
      },
    },
    Card: {
      backgroundColor: '#FFFFFF',
      borderRadius: designTokens.borderRadius.lg,
      boxShadow: designTokens.shadows.md,
      padding: designTokens.spacing.lg,
      transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.easeOut}`,

      '&:hover': {
        boxShadow: designTokens.shadows.lg,
        transform: 'translateY(-2px)',
      },
    },
    Timer: {
      fontSize: designTokens.typography.fontSize['2xl'], // Use 2xl for mobile fallback
      fontWeight: designTokens.typography.fontWeight.bold,
      color: designTokens.colors.neutral[500],
      fontFamily: designTokens.typography.fontFamily.secondary,

      // Responsive timer
      [designTokens.mediaQueries.tablet]: {
        fontSize: designTokens.typography.fontSize.tablet['3xl'],
      },

      [designTokens.mediaQueries.desktop]: {
        fontSize: designTokens.typography.fontSize.desktop['3xl'],
      },
    },
    Input: {
      backgroundColor: '#FFFFFF',
      border: `1px solid ${designTokens.colors.neutral[300]}`,
      borderRadius: designTokens.borderRadius.md,
      padding: `${designTokens.spacing.sm} ${designTokens.spacing.md}`,
      fontSize: designTokens.typography.fontSize.base,
      color: designTokens.colors.neutral[500],
      transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.easeInOut}`,

      '&:focus': {
        outline: 'none',
        borderColor: designTokens.colors.primary.main,
        boxShadow: `0 0 0 2px ${designTokens.colors.primary.light}33`,
      },

      '&::placeholder': {
        color: designTokens.colors.neutral[300],
      },
    },
    Navigation: {
      backgroundColor: '#FFFFFF',
      boxShadow: designTokens.shadows.sm,
      padding: `${designTokens.spacing.mobile.sm} 0`,
      position: 'sticky',
      top: '0',
      zIndex: 100,

      // Responsive navigation
      [designTokens.mediaQueries.mobile]: {
        padding: `${designTokens.spacing.mobile.xs} 0`,
        flexDirection: 'column',
        gap: designTokens.spacing.mobile.sm,
      },

      [designTokens.mediaQueries.tablet]: {
        padding: `${designTokens.spacing.tablet.sm} 0`,
        flexDirection: 'row',
        gap: designTokens.spacing.tablet.md,
      },

      [designTokens.mediaQueries.desktop]: {
        padding: `${designTokens.spacing.sm} 0`,
        flexDirection: 'row',
        gap: designTokens.spacing.md,
      },
    },
    ZenGarden: {
      backgroundColor: designTokens.colors.neutral[50],
      borderRadius: designTokens.borderRadius.lg,
      padding: designTokens.spacing.lg,
      minHeight: '200px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      position: 'relative',
      overflow: 'hidden',
    },
    ProgressRing: {
      transform: 'rotate(-90deg)',
      filter: 'drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1))',
    },
    TaskCard: {
      backgroundColor: '#FFFFFF',
      borderLeft: `4px solid ${designTokens.colors.primary.main}`,
      borderRadius: designTokens.borderRadius.md,
      boxShadow: designTokens.shadows.sm,
      padding: designTokens.spacing.mobile.md,
      marginBottom: designTokens.spacing.mobile.sm,
      transition: `all ${designTokens.animation.duration.normal} ${designTokens.animation.easing.easeOut}`,

      '&:hover': {
        boxShadow: designTokens.shadows.md,
        transform: 'translateX(4px)',
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
        borderLeftColor: designTokens.colors.error,
      },

      '&.medium-priority': {
        borderLeftColor: designTokens.colors.warning,
      },

      '&.low-priority': {
        borderLeftColor: designTokens.colors.success,
      },

      '&.completed': {
        borderLeftColor: designTokens.colors.success,
        opacity: 0.8,
      },
    },
    AchievementBadge: {
      backgroundColor: designTokens.colors.accent.light,
      color: '#FFFFFF',
      borderRadius: designTokens.borderRadius.full,
      padding: `${designTokens.spacing.sm} ${designTokens.spacing.md}`,
      fontSize: designTokens.typography.fontSize.sm,
      fontWeight: designTokens.typography.fontWeight.medium,
      display: 'inline-flex',
      alignItems: 'center',
      gap: designTokens.spacing.xs,
    },
    ProgressBar: {
      backgroundColor: designTokens.colors.neutral[200],
      borderRadius: designTokens.borderRadius.full,
      height: '8px',
      overflow: 'hidden',
      position: 'relative',

      '&::after': {
        content: '""',
        position: 'absolute',
        top: '0',
        left: '0',
        height: '100%',
        backgroundColor: designTokens.colors.primary.main,
        borderRadius: designTokens.borderRadius.full,
        transition: `width ${designTokens.animation.duration.normal} ${designTokens.animation.easing.easeInOut}`,
      },
    },
  },
};

// DefaultTheme will be automatically extended by the export below