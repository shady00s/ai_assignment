import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { useNavigate, useLocation } from 'react-router-dom';
import { useGetProfileQuery } from '@/store/api/apiSlice';

interface EnhancedNavigationProps {
  onMenuToggle?: () => void;
  onNotificationsClick?: () => void;
  onThemeToggle?: () => void;
  isDarkMode?: boolean;
  className?: string;
}

const NavContainer = styled.nav`
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  padding: ${({ theme }) => theme.spacing.mobile.sm} ${({ theme }) => theme.spacing.mobile.md};
  gap: ${({ theme }) => theme.spacing.mobile.md};

  /* Modern glassmorphism background */
  background: ${({ theme }) => theme.colors.glass.background};
  backdrop-filter: blur(${({ theme }) => theme.colors.glass.blur});
  border: 1px solid ${({ theme }) => theme.colors.glass.border};
  border-radius: ${({ theme }) => theme.borderRadius['2xl']};
  box-shadow: ${({ theme }) => theme.shadows.lg};

  /* Subtle gradient overlay */
  position: relative;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg,
      ${({ theme }) => theme.colors.primary[50]}20 0%,
      transparent 100%
    );
    border-radius: inherit;
    z-index: -1;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.sm} ${({ theme }) => theme.spacing.tablet.lg};
    gap: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.xl};
    gap: ${({ theme }) => theme.spacing.xl};
  }
`;

const LeftSection = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.md};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: ${({ theme }) => theme.spacing.lg};
  }
`;

const MenuButton = styled.button`
  background: transparent;
  border: none;
  color: ${({ theme }) => theme.colors.neutral[600]};
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  transition: all ${({ theme }) => theme.animation.duration.fast} ${({ theme }) => theme.animation.easing.smooth};
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;

  &:hover {
    background: ${({ theme }) => theme.colors.neutral[100]};
    color: ${({ theme }) => theme.colors.primary[600]};
  }

  &:active {
    background: ${({ theme }) => theme.colors.neutral[200]};
  }

  /* Hide menu button on desktop */
  ${({ theme }) => theme.mediaQueries.desktop} {
    display: none;
  }
`;

const HamburgerIcon = styled.div`
  width: 20px;
  height: 14px;
  position: relative;

  &::before,
  &::after {
    content: '';
    position: absolute;
    left: 0;
    width: 100%;
    height: 2px;
    background: currentColor;
    border-radius: 1px;
    transition: all ${({ theme }) => theme.animation.duration.fast} ${({ theme }) => theme.animation.easing.smooth};
  }

  &::before {
    top: 0;
  }

  &::after {
    bottom: 0;
  }

  span {
    position: absolute;
    top: 50%;
    left: 0;
    width: 100%;
    height: 2px;
    background: currentColor;
    border-radius: 1px;
    transform: translateY(-50%);
  }
`;

const AppTitle = styled.h1`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.h4};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.neutral[800]};
  margin: 0;
  font-family: ${({ theme }) => theme.typography.fontFamily.heading};
  background: ${({ theme }) => theme.colors.accent.gradient.ocean};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  background-size: 200% 200%;
  animation: gradient-shift 4s ease infinite;

  @keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.h3};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: ${({ theme }) => theme.typography.fontSize.h3};
  }
`;

const MainNavigation = styled.div`
  display: none;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  /* Show navigation only on desktop */
  ${({ theme }) => theme.mediaQueries.desktop} {
    display: flex;
    gap: ${({ theme }) => theme.spacing.md};
  }
`;

const NavButton = styled.button<{ $active: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing.mobile.sm} ${({ theme }) => theme.spacing.mobile.md};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  font-family: ${({ theme }) => theme.typography.fontFamily.body};

  border: none;
  background: transparent;
  color: ${({ $active, theme }) => $active
    ? theme.colors.primary[600]
    : theme.colors.neutral[600]};
  cursor: pointer;
  transition: all ${({ theme }) => theme.animation.duration.fast} ${({ theme }) => theme.animation.easing.smooth};
  white-space: nowrap;
  position: relative;

  /* Simple underline indicator for active state */
  ${({ $active, theme }) =>
    $active &&
    `
    &::after {
      content: '';
      position: absolute;
      bottom: -2px;
      left: 50%;
      transform: translateX(-50%);
      width: 20px;
      height: 2px;
      background: ${theme.colors.primary[600]};
      border-radius: 1px;
    }
  `}

  &:hover {
    color: ${({ theme }) => theme.colors.primary[700]};
    background: ${({ theme }) => theme.colors.neutral[50]};
  }

  &:active {
    color: ${({ theme }) => theme.colors.primary[800]};
    background: ${({ theme }) => theme.colors.neutral[100]};
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.sm} ${({ theme }) => theme.spacing.tablet.md};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
  }
`;

const RightSection = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: ${({ theme }) => theme.spacing.sm};
  }
`;

const IconButton = styled.button<{ $hasNotification?: boolean }>`
  position: relative;
  background: transparent;
  border: none;
  color: ${({ theme }) => theme.colors.neutral[600]};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.h4};
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  transition: all ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.smooth};
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;

  &:hover {
    background: ${({ theme }) => theme.colors.neutral[100]};
    color: ${({ theme }) => theme.colors.primary[600]};
    transform: translateY(-1px);
  }

  &:active {
    transform: translateY(0);
    background: ${({ theme }) => theme.colors.neutral[200]};
  }

  /* Modern notification indicator */
  ${({ $hasNotification, theme }) =>
    $hasNotification &&
    `
    &::after {
      content: '';
      position: absolute;
      top: 8px;
      right: 8px;
      width: 8px;
      height: 8px;
      background: ${theme.colors.error.main};
      border-radius: 50%;
      box-shadow: 0 0 0 2px ${theme.colors.glass.background};
    }
  `}

  ${({ theme }) => theme.mediaQueries.tablet} {
    width: 48px;
    height: 48px;
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.h4};
    padding: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    width: 52px;
    height: 52px;
    font-size: ${({ theme }) => theme.typography.fontSize.h4};
    padding: ${({ theme }) => theme.spacing.sm};
  }
`;

const NotificationBadge = styled.span`
  position: absolute;
  top: -2px;
  right: -2px;
  background: ${({ theme }) => theme.colors.error.main};
  color: white;
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  padding: 2px 5px;
  border-radius: ${({ theme }) => theme.borderRadius.full};
  min-width: 18px;
  text-align: center;
  border: 2px solid ${({ theme }) => theme.colors.glass.background};
  box-shadow: ${({ theme }) => theme.shadows.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.xs};
    padding: 3px 6px;
    min-width: 20px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    padding: 4px 7px;
    min-width: 22px;
  }
`;

const Divider = styled.div`
  width: 1px;
  height: 24px;
  background: ${({ theme }) => theme.colors.neutral[300]};
  border-radius: 1px;

  ${({ theme }) => theme.mediaQueries.tablet} {
    height: 28px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    height: 32px;
  }
`;

const ThemeToggleWrapper = styled.div`
  /* Hide theme toggle on mobile and tablet, show only on desktop */
  ${({ theme }) => theme.mediaQueries.mobile} {
    display: none;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    display: none;
  }
`;

const ProfileButton = styled.button`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  background: ${({ theme }) => theme.colors.glass.background};
  backdrop-filter: blur(${({ theme }) => theme.colors.glass.blur});
  border: 1px solid ${({ theme }) => theme.colors.glass.border};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  padding: ${({ theme }) => theme.spacing.mobile.sm} ${({ theme }) => theme.spacing.mobile.md};
  cursor: pointer;
  transition: all ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.smooth};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: ${({ theme }) => theme.colors.neutral[700]};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  font-family: ${({ theme }) => theme.typography.fontFamily.body};

  &:hover {
    transform: translateY(-1px);
    box-shadow: ${({ theme }) => theme.shadows.md};
    border-color: ${({ theme }) => theme.colors.primary[200]};
    color: ${({ theme }) => theme.colors.primary[600]};
  }

  &:active {
    transform: translateY(0);
  }

  /* Mobile: Show only avatar */
  ${({ theme }) => theme.mediaQueries.mobile} {
    padding: ${({ theme }) => theme.spacing.mobile.sm};
    gap: 0;
    background: transparent;
    border: none;
    backdrop-filter: none;

    &:hover {
      background: ${({ theme }) => theme.colors.neutral[100]};
      border-color: transparent;
      box-shadow: none;
    }
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.sm};
    padding: ${({ theme }) => theme.spacing.tablet.sm} ${({ theme }) => theme.spacing.tablet.md};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: ${({ theme }) => theme.spacing.sm};
    padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
  }
`;

const ProfileUsername = styled.span`
  /* Hide username on mobile */
  ${({ theme }) => theme.mediaQueries.mobile} {
    display: none;
  }
`;

const UserAvatar = styled.div`
  width: 32px;
  height: 32px;
  border-radius: ${({ theme }) => theme.borderRadius.full};
  background: ${({ theme }) => theme.colors.accent.gradient.aurora};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-family: ${({ theme }) => theme.typography.fontFamily.heading};
  box-shadow: ${({ theme }) => theme.shadows.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    width: 36px;
    height: 36px;
    font-size: ${({ theme }) => theme.typography.fontSize.base};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    width: 40px;
    height: 40px;
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
  }
`;

// Mobile Navigation Components
const MobileNavOverlay = styled.div<{ $isOpen: boolean }>`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
  z-index: 1000;
  opacity: ${({ $isOpen }) => ($isOpen ? 1 : 0)};
  visibility: ${({ $isOpen }) => ($isOpen ? 'visible' : 'hidden')};
  transition: all ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.smooth};

  /* Hide mobile navigation overlay on desktop */
  ${({ theme }) => theme.mediaQueries.desktop} {
    display: none;
  }
`;

const MobileNavDrawer = styled.div<{ $isOpen: boolean }>`
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  width: 280px;
  background: ${({ theme }) => theme.colors.neutral[50]};
  border: none;
  border-right: 1px solid ${({ theme }) => theme.colors.neutral[200]};
  box-shadow: ${({ theme }) => theme.shadows.xl};
  z-index: 1001;
  transform: translateX(${({ $isOpen }) => ($isOpen ? '0' : '-100%')});
  transition: transform ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.expo};
  display: flex;
  flex-direction: column;

  /* Hide mobile navigation drawer on desktop */
  ${({ theme }) => theme.mediaQueries.desktop} {
    display: none;
  }
`;

const MobileNavHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => theme.spacing.mobile.lg};
  border-bottom: 1px solid ${({ theme }) => theme.colors.neutral[200]};
`;

const MobileNavTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.h4};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.neutral[800]};
  margin: 0;
  font-family: ${({ theme }) => theme.typography.fontFamily.heading};
`;

const MobileNavCloseButton = styled.button`
  background: transparent;
  border: none;
  color: ${({ theme }) => theme.colors.neutral[600]};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.h3};
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  transition: all ${({ theme }) => theme.animation.duration.fast} ${({ theme }) => theme.animation.easing.smooth};
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;

  &:hover {
    background: ${({ theme }) => theme.colors.neutral[100]};
    color: ${({ theme }) => theme.colors.neutral[800]};
  }

  &:active {
    background: ${({ theme }) => theme.colors.neutral[200]};
  }
`;

const MobileNavContent = styled.div`
  flex: 1;
  padding: ${({ theme }) => theme.spacing.mobile.lg};
  overflow-y: auto;
`;

const MobileNavList = styled.ul`
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
`;

const MobileNavListItem = styled.li`
  margin: 0;
`;

const MobileNavLink = styled.button<{ $active: boolean }>`
  width: 100%;
  display: flex;
  align-items: center;
  padding: ${({ theme }) => theme.spacing.mobile.md};
  border: none;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  background: ${({ $active, theme }) =>
    $active ? theme.colors.primary[50] : 'transparent'};
  color: ${({ $active, theme }) =>
    $active ? theme.colors.primary[700] : theme.colors.neutral[700]};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.md};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  font-family: ${({ theme }) => theme.typography.fontFamily.body};
  text-align: left;
  cursor: pointer;
  transition: all ${({ theme }) => theme.animation.duration.fast} ${({ theme }) => theme.animation.easing.smooth};

  &:hover {
    background: ${({ theme }) => theme.colors.neutral[50]};
    color: ${({ theme }) => theme.colors.primary[600]};
    transform: translateX(4px);
  }

  &:active {
    background: ${({ theme }) => theme.colors.neutral[100]};
    transform: translateX(2px);
  }
`;

const MobileNavFooter = styled.div`
  padding: ${({ theme }) => theme.spacing.mobile.lg};
  border-top: 1px solid ${({ theme }) => theme.colors.neutral[200]};
  margin-top: auto;
`;

const MobileProfileButton = styled.button`
  width: 100%;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.md};
  background: ${({ theme }) => theme.colors.primary[50]};
  border: 1px solid ${({ theme }) => theme.colors.primary[200]};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing.mobile.md};
  cursor: pointer;
  transition: all ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.smooth};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: ${({ theme }) => theme.colors.primary[700]};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  font-family: ${({ theme }) => theme.typography.fontFamily.body};
  justify-content: flex-start;

  &:hover {
    background: ${({ theme }) => theme.colors.primary[100]};
    border-color: ${({ theme }) => theme.colors.primary[300]};
    transform: translateY(-1px);
    box-shadow: ${({ theme }) => theme.shadows.md};
  }

  &:active {
    transform: translateY(0);
  }
`;

export const EnhancedNavigation: React.FC<EnhancedNavigationProps> = ({
  onMenuToggle,
  onNotificationsClick,
  onThemeToggle,
  isDarkMode = false,
  className,
}) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isPressed, setIsPressed] = useState<string | null>(null);
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);
  const mobileNavRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  // Get user profile data
  const { data: userProfile } = useGetProfileQuery();

  const handlePress = (action: string, callback?: () => void) => {
    setIsPressed(action);
    setTimeout(() => setIsPressed(null), 150);
    callback?.();
  };

  const handleMobileNavToggle = () => {
    const newState = !isMobileNavOpen;
    setIsMobileNavOpen(newState);

    // Focus management
    if (newState && closeButtonRef.current) {
      setTimeout(() => closeButtonRef.current?.focus(), 100);
    }
  };

  const handleMobileNavClose = () => {
    setIsMobileNavOpen(false);
  };

  const handleMobileNavClick = (item: typeof navItems[0]) => {
    navigate(item.path);
    handleMobileNavClose();
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isMobileNavOpen) return;

      if (event.key === 'Escape') {
        handleMobileNavClose();
        return;
      }

      // Focus trapping within mobile nav
      if (event.key === 'Tab') {
        const mobileNav = mobileNavRef.current;
        if (!mobileNav) return;

        const focusableElements = mobileNav.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );

        const firstElement = focusableElements[0] as HTMLElement;
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

        if (event.shiftKey) {
          if (document.activeElement === firstElement) {
            event.preventDefault();
            lastElement?.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            event.preventDefault();
            firstElement?.focus();
          }
        }
      }
    };

    if (isMobileNavOpen) {
      document.addEventListener('keydown', handleKeyDown);
      // Prevent body scroll when mobile nav is open
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isMobileNavOpen]);

  // Navigation items without emoji icons
  const navItems = [
    {
      id: 'timer' as const,
      label: 'Timer',
      path: '/timer',
    },
    {
      id: 'tasks' as const,
      label: 'Tasks',
      path: '/tasks',
    },
    {
      id: 'dashboard' as const,
      label: 'Dashboard',
      path: '/dashboard',
    },
    {
      id: 'community' as const,
      label: 'Community',
      path: '/community',
    },
  ];

  const handleNavClick = (item: typeof navItems[0]) => {
    navigate(item.path);
  };

  // Determine active view based on current location
  const activeView = navItems.find(item => item.path === location.pathname)?.id || 'timer';

  const userName = userProfile?.firstName || userProfile?.email?.split('@')[0] || 'User';
  const notificationCount = 0; // TODO: Get from actual notifications

  return (
    <>
      <NavContainer className={className}>
        <LeftSection>
          <MenuButton
            onClick={() => handlePress('mobileMenu', handleMobileNavToggle)}
            aria-label="Menu"
            style={{
              transform: isPressed === 'mobileMenu' ? 'scale(0.95)' : 'scale(1)',
            }}
          >
            <HamburgerIcon>
              <span></span>
            </HamburgerIcon>
          </MenuButton>
          <AppTitle>OptoPomodoro</AppTitle>
        </LeftSection>

        <MainNavigation>
          {navItems.map((item) => (
            <NavButton
              key={item.id}
              onClick={() => handleNavClick(item)}
              $active={activeView === item.id}
              aria-label={item.label}
              title={item.label}
              style={{
                transform: isPressed === item.id ? 'scale(0.98)' : 'scale(1)',
              }}
            >
              {item.label}
            </NavButton>
          ))}
        </MainNavigation>

        <RightSection>
          <IconButton
            onClick={() => handlePress('notifications', onNotificationsClick)}
            $hasNotification={notificationCount > 0}
            aria-label="Notifications"
            title="Notifications"
            style={{
              transform: isPressed === 'notifications' ? 'scale(0.95)' : 'scale(1)',
            }}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0a1 1 0 0 1 1 1v1.07a7 7 0 0 1 3.993 3.993H14a1 1 0 1 1 0 2h-1.07a7 7 0 0 1-3.993 3.993V14a1 1 0 1 1-2 0v-1.07A7 7 0 0 1 3.007 9.007H2a1 1 0 0 1 0-2h1.07A7 7 0 0 1 6.993 3.007V2a1 1 0 0 1 1-1z"/>
              <circle cx="8" cy="8" r="3"/>
            </svg>
            {notificationCount > 0 && (
              <NotificationBadge>{notificationCount > 99 ? '99+' : notificationCount}</NotificationBadge>
            )}
          </IconButton>

          <Divider />

          <ThemeToggleWrapper>
            <IconButton
              onClick={() => handlePress('theme', onThemeToggle)}
              aria-label="Toggle theme"
              title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              style={{
                transform: isPressed === 'theme' ? 'scale(0.95)' : 'scale(1)',
              }}
            >
              {isDarkMode ? (
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                  <path d="M6 .278a.768.768 0 0 1 .08.858 7.208 7.208 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277.527 0 1.04-.055 1.533-.16a.787.787 0 0 1 .81.316.733.733 0 0 1-.031.893A8.349 8.349 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.752.752 0 0 1 6 .278z"/>
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                  <path d="M8 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm0 1a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM8 0a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 0zm0 13a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 13zm8-5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2a.5.5 0 0 1 .5.5zM3 8a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2A.5.5 0 0 1 3 8zm10.657-5.657a.5.5 0 0 1 0 .707l-1.414 1.415a.5.5 0 1 1-.707-.708l1.414-1.414a.5.5 0 0 1 .707 0zm-9.193 9.193a.5.5 0 0 1 0 .707L3.05 13.657a.5.5 0 0 1-.707-.707l1.414-1.414a.5.5 0 0 1 .707 0zm9.193 2.121a.5.5 0 0 1-.707 0l-1.414-1.414a.5.5 0 0 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .707zM4.464 4.465a.5.5 0 0 1-.707 0L2.343 3.05a.5.5 0 1 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .708z"/>
                </svg>
              )}
            </IconButton>
          </ThemeToggleWrapper>

  
          <ProfileButton
            onClick={() => {
              handlePress('userMenu', () => navigate('/profile'));
            }}
            aria-label="User profile"
            title="Go to profile"
            style={{
              transform: isPressed === 'userMenu' ? 'scale(0.98)' : 'scale(1)',
            }}
          >
            <UserAvatar>
              {userName.charAt(0).toUpperCase()}
            </UserAvatar>
            <ProfileUsername>{userName}</ProfileUsername>
          </ProfileButton>
        </RightSection>
      </NavContainer>

      {/* Mobile Navigation */}
      <MobileNavOverlay
        $isOpen={isMobileNavOpen}
        onClick={handleMobileNavClose}
        aria-hidden={!isMobileNavOpen}
      />
      <MobileNavDrawer ref={mobileNavRef} $isOpen={isMobileNavOpen} aria-hidden={!isMobileNavOpen}>
        <MobileNavHeader>
          <MobileNavTitle>OptoPomodoro</MobileNavTitle>
          <MobileNavCloseButton
            ref={closeButtonRef}
            onClick={handleMobileNavClose}
            aria-label="Close menu"
          >
            ×
          </MobileNavCloseButton>
        </MobileNavHeader>

        <MobileNavContent>
          <MobileNavList>
            {navItems.map((item) => (
              <MobileNavListItem key={item.id}>
                <MobileNavLink
                  onClick={() => handleMobileNavClick(item)}
                  $active={activeView === item.id}
                >
                  {item.label}
                </MobileNavLink>
              </MobileNavListItem>
            ))}
          </MobileNavList>

          {/* Theme Toggle Section */}
          <div style={{ marginTop: '24px' }}>
            <div style={{
              fontSize: '0.875rem',
              fontWeight: '600',
              color: '#64748b',
              marginBottom: '12px',
              fontFamily: '"Inter Variable", sans-serif'
            }}>
              Settings
            </div>
            <MobileNavLink
              onClick={() => {
                onThemeToggle?.();
                handleMobileNavClose();
              }}
              $active={false}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <span>Theme</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{
                  fontSize: '0.75rem',
                  color: '#64748b',
                  fontFamily: '"Inter Variable", sans-serif'
                }}>
                  {isDarkMode ? 'Dark' : 'Light'}
                </span>
                {isDarkMode ? (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M6 .278a.768.768 0 0 1 .08.858 7.208 7.208 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277.527 0 1.04-.055 1.533-.16a.787.787 0 0 1 .81.316.733.733 0 0 1-.031.893A8.349 8.349 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.752.752 0 0 1 6 .278z"/>
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm0 1a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM8 0a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 0zm0 13a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 13zm8-5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2a.5.5 0 0 1 .5.5zM3 8a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2A.5.5 0 0 1 3 8zm10.657-5.657a.5.5 0 0 1 0 .707l-1.414 1.415a.5.5 0 1 1-.707-.708l1.414-1.414a.5.5 0 0 1 .707 0zm-9.193 9.193a.5.5 0 0 1 0 .707L3.05 13.657a.5.5 0 0 1-.707-.707l1.414-1.414a.5.5 0 0 1 .707 0zm9.193 2.121a.5.5 0 0 1-.707 0l-1.414-1.414a.5.5 0 0 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .707zM4.464 4.465a.5.5 0 0 1-.707 0L2.343 3.05a.5.5 0 1 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .708z"/>
                  </svg>
                )}
              </div>
            </MobileNavLink>
          </div>
        </MobileNavContent>

        <MobileNavFooter>
          <MobileProfileButton
            onClick={() => {
              handleMobileNavClose();
              navigate('/profile');
            }}
          >
            <UserAvatar>
              {userName.charAt(0).toUpperCase()}
            </UserAvatar>
            <span>{userName}</span>
          </MobileProfileButton>
        </MobileNavFooter>
      </MobileNavDrawer>
    </>
  );
};

export type { EnhancedNavigationProps };