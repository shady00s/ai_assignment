import { useState, useEffect } from 'react';

interface ResponsiveLayoutOptions {
  mobileBreakpoint?: number;
  tabletBreakpoint?: number;
}

type Breakpoint = 'mobile' | 'tablet' | 'desktop';

interface UseResponsiveLayoutReturn {
  breakpoint: Breakpoint;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  windowWidth: number;
}

export const useResponsiveLayout = (options: ResponsiveLayoutOptions = {}): UseResponsiveLayoutReturn => {
  const { mobileBreakpoint = 768, tabletBreakpoint = 1024 } = options;

  const [windowWidth, setWindowWidth] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.innerWidth;
    }
    return 1200; // Default desktop width
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Set initial width

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  const getBreakpoint = (width: number): Breakpoint => {
    if (width < mobileBreakpoint) return 'mobile';
    if (width < tabletBreakpoint) return 'tablet';
    return 'desktop';
  };

  const breakpoint = getBreakpoint(windowWidth);
  const isMobile = breakpoint === 'mobile';
  const isTablet = breakpoint === 'tablet';
  const isDesktop = breakpoint === 'desktop';

  return {
    breakpoint,
    isMobile,
    isTablet,
    isDesktop,
    windowWidth,
  };
};