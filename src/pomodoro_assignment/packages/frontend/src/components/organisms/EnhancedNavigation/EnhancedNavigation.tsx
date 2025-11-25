import React, { useState } from 'react';
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
  padding: 0;
  gap: ${({ theme }) => theme.spacing.mobile.md};
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(15px);
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(127, 168, 112, 0.1);

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
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
  background: none;
  border: none;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    background: rgba(127, 168, 112, 0.1);
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.xl};
    padding: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: ${({ theme }) => theme.typography.fontSize.xl};
    padding: ${({ theme }) => theme.spacing.sm};
  }
`;

const AppTitle = styled.h1`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: #2C3E50;
  margin: 0;
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: ${({ theme }) => theme.typography.fontSize.xl};
  }
`;

const MainNavigation = styled.div`
  display: none;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    display: flex;
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    display: flex;
    gap: ${({ theme }) => theme.spacing.sm};
  }
`;

const NavButton = styled.button<{ $active: boolean; $color: string }>`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.mobile.xs};
  padding: ${({ theme }) => theme.spacing.mobile.sm} ${({ theme }) => theme.spacing.mobile.lg};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  border: ${({ $active, $color }) => $active ? 'none' : `2px solid ${$color}20`};
  background: ${({ $active, $color }) =>
    $active
      ? `linear-gradient(135deg, ${$color} 0%, ${$color}DD 100%)`
      : 'rgba(255, 255, 255, 0.8)'};
  color: ${({ $active, $color }) => $active ? 'white' : $color};
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  white-space: nowrap;
  box-shadow: ${({ $active, $color }) =>
    $active
      ? `0 4px 12px ${$color}40, 0 2px 4px ${$color}20`
      : `0 2px 8px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.06)`};
  backdrop-filter: blur(10px);
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 100%);
    opacity: ${({ $active }) => $active ? 1 : 0};
    transition: opacity 0.3s ease;
  }

  &:hover {
    background: ${({ $active, $color }) =>
      $active
        ? `linear-gradient(135deg, ${$color} 0%, ${$color}EE 100%)`
        : `${$color}10`};
    transform: translateY(-2px);
    box-shadow: ${({ $active, $color }) =>
      $active
        ? `0 8px 20px ${$color}50, 0 4px 8px ${$color}30`
        : `0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.04)`};
    border-color: ${({ $color }) => $color}40;
  }

  &:active {
    transform: translateY(-1px);
    box-shadow: ${({ $active, $color }) =>
      $active
        ? `0 4px 12px ${$color}40, 0 2px 4px ${$color}20`
        : `0 2px 6px rgba(0, 0, 0, 0.06), 0 1px 3px rgba(0, 0, 0, 0.04)`};
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.sm} ${({ theme }) => theme.spacing.tablet.lg};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
    gap: ${({ theme }) => theme.spacing.tablet.sm};
    border-radius: ${({ theme }) => theme.borderRadius['2xl']};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.xl};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    gap: ${({ theme }) => theme.spacing.sm};
    border-radius: 16px;
  }
`;

const NavIcon = styled.span`
  font-size: 16px;
  line-height: 1;
  position: relative;
  z-index: 1;

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 18px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 20px;
  }
`;

const RightSection = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: ${({ theme }) => theme.spacing.md};
  }
`;

const IconButton = styled.button<{ $hasNotification?: boolean }>`
  position: relative;
  background: none;
  border: none;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.md};
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  border-radius: 50%;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;

  &:hover {
    background: rgba(127, 168, 112, 0.1);
    transform: translateY(-1px);
  }

  ${({ $hasNotification }) =>
    $hasNotification &&
    `
    &::after {
      content: '';
      position: absolute;
      top: 4px;
      right: 4px;
      width: 8px;
      height: 8px;
      background: #C85A5A;
      border-radius: 50%;
      border: 2px solid white;
    }
  `}

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.lg};
    width: 44px;
    height: 44px;
    padding: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
    width: 48px;
    height: 48px;
    padding: ${({ theme }) => theme.spacing.sm};
  }
`;

const NotificationBadge = styled.span`
  position: absolute;
  top: -2px;
  right: -2px;
  background: #C85A5A;
  color: white;
  font-size: 10px;
  font-weight: bold;
  padding: 2px 4px;
  border-radius: 10px;
  min-width: 16px;
  text-align: center;
  border: 2px solid white;

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 11px;
    padding: 3px 5px;
    min-width: 18px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 12px;
    padding: 4px 6px;
    min-width: 20px;
  }
`;

const ProfileButton = styled.button`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  background: rgba(127, 168, 112, 0.1);
  border: 1px solid rgba(127, 168, 112, 0.2);
  border-radius: 20px;
  padding: ${({ theme }) => theme.spacing.mobile.sm} ${({ theme }) => theme.spacing.mobile.md};
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #2C3E50;
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(127, 168, 112, 0.2);
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.sm};
    padding: ${({ theme }) => theme.spacing.tablet.sm} ${({ theme }) => theme.spacing.tablet.md};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
    border-radius: 24px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: ${({ theme }) => theme.spacing.sm};
    padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    border-radius: 28px;
  }
`;

const UserAvatar = styled.div`
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
  font-size: 12px;

  ${({ theme }) => theme.mediaQueries.tablet} {
    width: 32px;
    height: 32px;
    font-size: 14px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    width: 36px;
    height: 36px;
    font-size: 16px;
  }
`;

const Divider = styled.div`
  width: 1px;
  height: 24px;
  background: rgba(127, 168, 112, 0.2);

  ${({ theme }) => theme.mediaQueries.tablet} {
    height: 28px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    height: 32px;
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

  // Get user profile data
  const { data: userProfile } = useGetProfileQuery();

  const handlePress = (action: string, callback?: () => void) => {
    setIsPressed(action);
    setTimeout(() => setIsPressed(null), 150);
    callback?.();
  };

  // Navigation items
  const navItems = [
    {
      id: 'timer' as const,
      label: 'Timer',
      icon: '⏰',
      color: '#7FA870',
      path: '/timer',
    },
    {
      id: 'tasks' as const,
      label: 'Tasks',
      icon: '📋',
      color: '#F4A261',
      path: '/tasks',
    },
    {
      id: 'dashboard' as const,
      label: 'Dashboard',
      icon: '📊',
      color: '#6B8E9F',
      path: '/dashboard',
    },
    {
      id: 'community' as const,
      label: 'Community',
      icon: '👥',
      color: '#E9C46A',
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
    <NavContainer className={className}>
      <LeftSection>
        <MenuButton
          onClick={() => handlePress('menu', onMenuToggle)}
          aria-label="Menu"
          style={{
            transform: isPressed === 'menu' ? 'scale(0.95)' : 'scale(1)',
          }}
        >
          ☰
        </MenuButton>
        <AppTitle>OptoPomodoro</AppTitle>
      </LeftSection>

      <MainNavigation>
        {navItems.map((item) => (
          <NavButton
            key={item.id}
            onClick={() => handleNavClick(item)}
            $active={activeView === item.id}
            $color={item.color}
            aria-label={item.label}
            title={item.label}
            style={{
              transform: isPressed === item.id ? 'scale(0.98)' : 'scale(1)',
            }}
          >
            <NavIcon>{item.icon}</NavIcon>
            <span>{item.label}</span>
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
          🔔
          {notificationCount > 0 && (
            <NotificationBadge>{notificationCount > 99 ? '99+' : notificationCount}</NotificationBadge>
          )}
        </IconButton>

        <Divider />

        <IconButton
          onClick={() => handlePress('theme', onThemeToggle)}
          aria-label="Toggle theme"
          title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          style={{
            transform: isPressed === 'theme' ? 'scale(0.95)' : 'scale(1)',
          }}
        >
          {isDarkMode ? '🌙' : '🔆'}
        </IconButton>

        <ProfileButton
          onClick={() => {
            handlePress('userMenu', () => navigate('/profile'));
          }}
          aria-label="User profile"
          title="Go to profile"
          style={{
            transform: isPressed === 'userMenu' ? 'scale(0.98)' : 'scale(1)',
            background: location.pathname === '/profile'
              ? 'linear-gradient(135deg, #C85A5A 0%, #D57A7A 100%)'
              : undefined,
            color: location.pathname === '/profile' ? 'white' : '#2C3E50',
          }}
        >
          <UserAvatar>
            {userName.charAt(0).toUpperCase()}
          </UserAvatar>
          <span>{userName}</span>
        </ProfileButton>
      </RightSection>
    </NavContainer>
  );
};

export type { EnhancedNavigationProps };