export const designTokens = {
  // Modern breakpoints following industry standards
  breakpoints: {
    mobile: {
      xs: '320px',   // Small phones
      sm: '375px',   // Large phones
      md: '425px',   // Extra large phones
    },
    tablet: {
      sm: '640px',   // Small tablets
      md: '768px',   // Medium tablets
      lg: '1024px',  // Large tablets
    },
    desktop: {
      sm: '1280px',  // Small desktops
      md: '1440px',  // Medium desktops
      lg: '1920px',  // Large desktops
      xl: '2560px',  // Extra large desktops
    },
  },

  // Media queries for easy use
  mediaQueries: {
    mobile: `@media (max-width: ${425}px)`,
    tablet: `@media (min-width: 426px) and (max-width: 1023px)`,
    desktop: `@media (min-width: 1024px)`,
    // Mobile-first approach
    mobileUp: `@media (min-width: 320px)`,
    tabletUp: `@media (min-width: 768px)`,
    desktopUp: `@media (min-width: 1024px)`,
    largeDesktopUp: `@media (min-width: 1440px)`,
    extraLargeDesktopUp: `@media (min-width: 1920px)`,
  },

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
      xs: '10px',
      sm: '12px',
      base: '14px',
      lg: '16px',
      xl: '18px',
      '2xl': '24px',
      '3xl': '32px',
      '4xl': '40px',
      '5xl': '48px',
      '6xl': '64px',
      // Responsive font sizes
      mobile: {
        xs: '10px',
        sm: '12px',
        base: '14px',
        lg: '16px',
      },
      tablet: {
        xs: '12px',
        sm: '14px',
        base: '16px',
        lg: '18px',
        xl: '20px',
        '2xl': '24px',
        '3xl': '32px',
      },
      desktop: {
        xs: '12px',
        sm: '14px',
        base: '16px',
        lg: '18px',
        xl: '20px',
        '2xl': '24px',
        '3xl': '32px',
        '4xl': '40px',
        '5xl': '48px',
      },
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
    lg: '24px',
    xl: '32px',
    xxl: '48px',
    // Responsive spacing
    mobile: {
      xs: '4px',
      sm: '8px',
      md: '12px',
      lg: '16px',
    },
    tablet: {
      xs: '6px',
      sm: '10px',
      md: '16px',
      lg: '24px',
      xl: '32px',
    },
    desktop: {
      xs: '8px',
      sm: '12px',
      md: '16px',
      lg: '24px',
      xl: '32px',
      xxl: '48px',
    },
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