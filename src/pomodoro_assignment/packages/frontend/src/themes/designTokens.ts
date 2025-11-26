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
    // Modern Startup Primary Palette
    primary: {
      50: '#F0F9FF',        // Ghost White
      100: '#E0F2FE',       // Light Blue
      200: '#BAE6FD',       // Sky Blue Light
      300: '#7DD3FC',       // Sky Blue
      400: '#38BDF8',       // Bright Blue
      500: '#0EA5E9',       // Modern Blue
      600: '#0284C7',       // Ocean Blue
      700: '#0369A1',       // Deep Blue
      800: '#075985',       // Navy Blue
      900: '#0C4A6E',       // Midnight Blue
      main: '#0EA5E9',      // Primary Modern Blue
      light: '#7DD3FC',
      dark: '#0369A1',
    },
    // Startup Energy Secondary
    secondary: {
      50: '#FDF4FF',        // Light Purple
      100: '#FAE8FF',       // Lavender Light
      200: '#F5D0FE',       // Lavender
      300: '#F0ABFC',       // Bright Lavender
      400: '#E879F9',       // Electric Purple
      500: '#D946EF',       // Vibrant Purple
      600: '#C026D3',       // Deep Purple
      700: '#A21CAF',       // Rich Purple
      800: '#86198F',       // Dark Purple
      900: '#701A75',       // Midnight Purple
      main: '#D946EF',      // Startup Purple
      light: '#F0ABFC',
      dark: '#A21CAF',
    },
    // Modern Gradient Accents
    accent: {
      gradient: {
        sunrise: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        ocean: 'linear-gradient(135deg, #0EA5E9 0%, #D946EF 100%)',
        forest: 'linear-gradient(135deg, #10B981 0%, #3B82F6 100%)',
        sunset: 'linear-gradient(135deg, #F59E0B 0%, #EF4444 100%)',
        aurora: 'linear-gradient(135deg, #06FFB4 0%, #0EA5E9 100%)',
      },
      main: '#F59E0B',      // Modern Amber
      light: '#FCD34D',
      dark: '#D97706',
      neon: {
        pink: '#FF10F0',
        blue: '#00D9FF',
        green: '#39FF14',
        purple: '#BF00FF',
        yellow: '#FFFF00',
      }
    },
    // Modern Neutral Palette
    neutral: {
      0: '#FFFFFF',         // Pure White
      50: '#F8FAFC',        // Ghost White
      100: '#F1F5F9',       // Light Gray
      200: '#E2E8F0',       // Silver Light
      300: '#CBD5E1',       // Silver
      400: '#94A3B8',       // Slate Gray
      500: '#64748B',       // Medium Gray
      600: '#475569',       // Slate
      700: '#334155',       // Dark Slate
      800: '#1E293B',       // Charcoal
      900: '#0F172A',       // Midnight
      950: '#020617',       // Abyss
    },
    // Success with Modern Green
    success: {
      50: '#F0FDF4',        // Light Green
      100: '#DCFCE7',       // Mint Light
      200: '#BBF7D0',       // Mint
      300: '#86EFAC',       // Bright Green
      400: '#4ADE80',       // Success Green
      500: '#22C55E',       // Modern Green
      600: '#16A34A',       // Forest Green
      700: '#15803D',       // Dark Green
      800: '#166534',       // Deep Forest
      900: '#14532D',       // Midnight Forest
      main: '#22C55E',      // Primary Success
    },
    // Warning with Energy
    warning: {
      50: '#FFFBEB',        // Light Yellow
      100: '#FEF3C7',       // Cream Yellow
      200: '#FDE68A',       // Light Amber
      300: '#FCD34D',       // Bright Yellow
      400: '#FBBF24',       // Warning Yellow
      500: '#F59E0B',       // Modern Amber
      600: '#D97706',       // Deep Amber
      700: '#B45309',       // Dark Amber
      800: '#92400E',       // Brown
      900: '#78350F',       // Dark Brown
      main: '#F59E0B',      // Primary Warning
    },
    // Error with Modern Red
    error: {
      50: '#FEF2F2',        // Light Red
      100: '#FEE2E2',       // Rose Light
      200: '#FECACA',       // Rose
      300: '#FCA5A5',       // Bright Rose
      400: '#F87171',       // Error Red
      500: '#EF4444',       // Modern Red
      600: '#DC2626',       // Deep Red
      700: '#B91C1C',       // Dark Red
      800: '#991B1B',       // Very Dark Red
      900: '#7F1D1D',       // Midnight Red
      main: '#EF4444',      // Primary Error
    },
    // Info with Modern Cyan
    info: {
      50: '#ECFEFF',        // Light Cyan
      100: '#CFFAFE',       // Cyan Light
      200: '#A5F3FC',       // Cyan
      300: '#67E8F9',       // Bright Cyan
      400: '#22D3EE',       // Info Cyan
      500: '#06B6D4',       // Modern Cyan
      600: '#0891B2',       // Deep Cyan
      700: '#0E7490',       // Dark Cyan
      800: '#155E75',       // Very Dark Cyan
      900: '#164E63',       // Midnight Cyan
      main: '#06B6D4',      // Primary Info
    },
    // Dark Mode Specific Colors
    dark: {
      background: '#0F172A',     // Midnight background
      surface: '#1E293B',        // Charcoal surface
      card: '#334155',           // Dark slate card
      border: '#475569',         // Slate border
      text: '#F1F5F9',           // Light gray text
      textSecondary: '#94A3B8',  // Slate gray secondary text
      accent: '#D946EF',         // Vibrant purple accent
    },
    // Glassmorphism Effects
    glass: {
      background: 'rgba(255, 255, 255, 0.1)',
      border: 'rgba(255, 255, 255, 0.2)',
      backdrop: 'rgba(0, 0, 0, 0.05)',
      blur: '12px',
    },
  },

  typography: {
    fontFamily: {
      // Modern Startup Font Stack
      display: '"Inter Variable", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      heading: '"Space Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      body: '"Inter Variable", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      mono: '"JetBrains Mono", "Fira Code", Consolas, monospace',
      accent: '"Space Mono", monospace',
    },
    fontSize: {
      // Modern Scale with better readability
      xs: '0.75rem',      // 12px
      sm: '0.875rem',     // 14px
      base: '1rem',       // 16px
      lg: '1.125rem',     // 18px
      xl: '1.25rem',      // 20px
      '2xl': '1.5rem',    // 24px
      '3xl': '1.875rem',  // 30px
      '4xl': '2.25rem',   // 36px
      '5xl': '3rem',      // 48px
      '6xl': '3.75rem',   // 60px
      '7xl': '4.5rem',    // 72px
      '8xl': '6rem',      // 96px
      '9xl': '8rem',      // 128px

      // Semantic Font Sizes
      'hero': '4.5rem',           // 72px - Hero headings
      'display': '3.75rem',       // 60px - Display headings
      'h1': '3rem',               // 48px - Main headings
      'h2': '2.25rem',            // 36px - Section headings
      'h3': '1.875rem',           // 30px - Subsection headings
      'h4': '1.5rem',             // 24px - Component headings
      'h5': '1.25rem',            // 20px - Small headings
      'h6': '1.125rem',           // 18px - Micro headings
      'body-lg': '1.125rem',      // 18px - Large body text
      'body': '1rem',             // 16px - Normal body text
      'body-sm': '0.875rem',      // 14px - Small body text
      'body-xs': '0.75rem',       // 12px - Extra small body text
      'caption': '0.75rem',       // 12px - Caption text
      'label': '0.875rem',        // 14px - Form labels
      'overline': '0.75rem',      // 12px - Overline text

      // Responsive font sizes with better scaling
      mobile: {
        'hero': '2.5rem',         // 40px
        'display': '2.25rem',     // 36px
        'h1': '2rem',             // 32px
        'h2': '1.75rem',          // 28px
        'h3': '1.5rem',           // 24px
        'h4': '1.25rem',          // 20px
        'h5': '1.125rem',         // 18px
        'h6': '1rem',             // 16px
        'body-lg': '1rem',        // 16px
        'body': '0.9375rem',      // 15px
        'body-sm': '0.875rem',    // 14px
        'body-xs': '0.75rem',     // 12px
        'caption': '0.75rem',     // 12px
        'label': '0.875rem',      // 14px
        'overline': '0.75rem',    // 12px
      },
      tablet: {
        'hero': '3.5rem',         // 56px
        'display': '3rem',        // 48px
        'h1': '2.5rem',           // 40px
        'h2': '2rem',             // 32px
        'h3': '1.75rem',          // 28px
        'h4': '1.5rem',           // 24px
        'h5': '1.25rem',          // 20px
        'h6': '1.125rem',         // 18px
        'body-lg': '1.125rem',    // 18px
        'body': '1rem',           // 16px
        'body-sm': '0.875rem',    // 14px
        'body-xs': '0.75rem',     // 12px
        'caption': '0.75rem',     // 12px
        'label': '0.875rem',      // 14px
        'overline': '0.75rem',    // 12px
      },
      desktop: {
        'hero': '4.5rem',         // 72px
        'display': '3.75rem',     // 60px
        'h1': '3rem',             // 48px
        'h2': '2.25rem',          // 36px
        'h3': '1.875rem',         // 30px
        'h4': '1.5rem',           // 24px
        'h5': '1.25rem',          // 20px
        'h6': '1.125rem',         // 18px
        'body-lg': '1.125rem',    // 18px
        'body': '1rem',           // 16px
        'body-sm': '0.875rem',    // 14px
        'body-xs': '0.75rem',     // 12px
        'caption': '0.75rem',     // 12px
        'label': '0.875rem',      // 14px
        'overline': '0.75rem',    // 12px
      },
    },
    fontWeight: {
      // Modern font weights with better granularity
      thin: 100,
      extralight: 200,
      light: 300,
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
      extrabold: 800,
      black: 900,
    },
    lineHeight: {
      // Better line height scale for readability
      none: 1,
      tight: 1.25,
      snug: 1.375,
      normal: 1.5,
      relaxed: 1.625,
      loose: 2,
    },
    letterSpacing: {
      tighter: '-0.05em',
      tight: '-0.025em',
      normal: '0em',
      wide: '0.025em',
      wider: '0.05em',
      widest: '0.1em',
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
    none: '0',
    sm: '4px',
    md: '8px',
    lg: '16px',
    xl: '24px',
    '2xl': '32px',
    '3xl': '48px',
    full: '50%',
  },

  shadows: {
    // Modern shadow system with colored shadows
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
    // Colored shadows for startup vibe
    primary: '0 10px 25px -5px rgba(14, 165, 233, 0.25)',
    secondary: '0 10px 25px -5px rgba(217, 70, 239, 0.25)',
    success: '0 10px 25px -5px rgba(34, 197, 94, 0.25)',
    warning: '0 10px 25px -5px rgba(245, 158, 11, 0.25)',
    error: '0 10px 25px -5px rgba(239, 68, 68, 0.25)',
    // Glow effects
    glow: {
      primary: '0 0 20px rgba(14, 165, 233, 0.5)',
      secondary: '0 0 20px rgba(217, 70, 239, 0.5)',
      success: '0 0 20px rgba(34, 197, 94, 0.5)',
      warning: '0 0 20px rgba(245, 158, 11, 0.5)',
      error: '0 0 20px rgba(239, 68, 68, 0.5)',
    },
  },

  animation: {
    duration: {
      fast: '150ms',
      normal: '300ms',
      slow: '500ms',
      slower: '750ms',
      slowest: '1000ms',
    },
    easing: {
      easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
      // Modern easing functions
      bouncy: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
      smooth: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)',
      sharp: 'cubic-bezier(0.4, 0, 0.6, 1)',
      expo: 'cubic-bezier(0.19, 1, 0.22, 1)',
    },
    // Modern animation presets
    presets: {
      fadeIn: 'fadeIn 0.3s ease-out',
      slideUp: 'slideUp 0.3s ease-out',
      slideDown: 'slideDown 0.3s ease-out',
      slideLeft: 'slideLeft 0.3s ease-out',
      slideRight: 'slideRight 0.3s ease-out',
      scaleIn: 'scaleIn 0.2s ease-out',
      bounce: 'bounce 0.6s ease-out',
      pulse: 'pulse 2s infinite',
      spin: 'spin 1s linear infinite',
      ping: 'ping 1s cubic-bezier(0, 0, 0.2, 1) infinite',
      float: 'float 3s ease-in-out infinite',
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