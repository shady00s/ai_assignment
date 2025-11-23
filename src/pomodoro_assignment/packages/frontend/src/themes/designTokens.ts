export const designTokens = {
  colors: {
    primary: {
      main: '#7A8B7F',      // Moss Green
      light: '#9AA895',
      dark: '#5F6E63',
    },
    secondary: {
      main: '#6B8E9F',      // Water Blue
      light: '#8DA4B1',
      dark: '#537281',
    },
    accent: {
      main: '#E67E50',      // Sunrise Orange
      light: '#EB9D7A',
      dark: '#C46441',
    },
    neutral: {
      50: '#F4E4D4',        // Warm Sand
      100: '#E8D8C8',
      200: '#D4C5B9',      // Zen Stone
      300: '#C0B2A5',
      400: '#8B7D7B',      // Stone Gray
      500: '#2C3E50',      // Charcoal
    },
    success: '#7FA870',     // Sage Green
    warning: '#F4A261',     // Warm Amber
    error: '#C85A5A',       // Soft Red
    info: '#6B8E9F',        // Sky Blue
  },

  typography: {
    fontFamily: {
      primary: 'Inter, sans-serif',
      secondary: 'Lora, serif',
    },
    fontSize: {
      xs: '12px',
      sm: '14px',
      base: '16px',
      lg: '18px',
      xl: '24px',
      '2xl': '32px',
      '3xl': '48px',
    },
    fontWeight: {
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.75,
    },
  },

  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '32px',
    xl: '64px',
  },

  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '16px',
    full: '50%',
  },

  shadows: {
    sm: '0 1px 3px rgba(0, 0, 0, 0.1)',
    md: '0 4px 6px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 25px rgba(0, 0, 0, 0.1)',
  },

  animation: {
    duration: {
      fast: '150ms',
      normal: '300ms',
      slow: '500ms',
    },
    easing: {
      easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    },
  },
};

// Type definitions for the design tokens
export type ColorKeys = keyof typeof designTokens.colors;
export type FontFamilyKeys = keyof typeof designTokens.typography.fontFamily;
export type FontSizeKeys = keyof typeof designTokens.typography.fontSize;
export type FontWeightKeys = keyof typeof designTokens.typography.fontWeight;
export type SpacingKeys = keyof typeof designTokens.spacing;
export type BorderRadiusKeys = keyof typeof designTokens.borderRadius;
export type AnimationDurationKeys = keyof typeof designTokens.animation.duration;
export type AnimationEasingKeys = keyof typeof designTokens.animation.easing;