import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { Button } from '../../atoms/Button';

const NavContainer = styled.nav`
  width: 100%;
  padding: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme }) => theme.spacing.lg};
  }
`;

const NavContent = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 8px;
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  background-color: white;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  box-shadow: ${({ theme }) => theme.shadows.md};
  border: 1px solid rgba(127, 168, 112, 0.1);

  ${({ theme }) => theme.mediaQueries.tablet} {
    flex-direction: row;
    gap: ${({ theme }) => theme.spacing.tablet.sm};
    padding: ${({ theme }) => theme.spacing.tablet.md};
    max-width: 800px;
    margin: 0 auto;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    flex-direction: row;
    gap: ${({ theme }) => theme.spacing.sm};
    padding: ${({ theme }) => theme.spacing.md};
    max-width: 1000px;
    margin: 0 auto;
  }
`;

const NavButton = styled(Button)<{ $active: boolean; $color: string }>`
  flex: 1 1 auto;
  min-width: 50px;
  padding: 4px 6px;
  font-size: 9px;
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  transition: all 0.2s ease;
  height: 54px;
  text-align: center;
  line-height: 1.1;

  /* Very small screens (320px and below) */
  @media (max-width: 320px) {
    flex: 1 1 0%;
    min-width: 45px;
    padding: 3px 4px;
    font-size: 8px;
    height: 48px;
    gap: 1px;

    /* Adjust layout for inactive buttons (icon-only) */
    &:not([data-active="true"]) {
      justify-content: center;
      padding: 6px;
    }

    /* Adjust layout for active button (icon + text) */
    &[data-active="true"] {
      justify-content: center;
      gap: 1px;
    }
  }

  /* Small screens (321px - 425px) */
  ${({ theme }) => theme.mediaQueries.mobile} {
    flex: 1 1 auto;
    min-width: 55px;
    padding: 6px 8px;
    font-size: 10px;
    height: 55px;
    gap: 3px;

    /* Adjust layout for inactive buttons (icon-only) */
    &:not([data-active="true"]) {
      justify-content: center;
      padding: 8px;
    }

    /* Adjust layout for active button (icon + text) */
    &[data-active="true"] {
      justify-content: center;
      gap: 2px;
    }
  }

  /* Tablet screens */
  ${({ theme }) => theme.mediaQueries.tablet} {
    flex: 1 1 0%;
    min-width: 80px;
    padding: ${({ theme }) => theme.spacing.tablet.sm} ${({ theme }) => theme.spacing.tablet.sm};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
    flex-direction: row;
    gap: ${({ theme }) => theme.spacing.tablet.xs};
    height: auto;
    line-height: 1.3;
  }

  /* Desktop screens */
  ${({ theme }) => theme.mediaQueries.desktop} {
    flex: none;
    min-width: 120px;
    padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.lg};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
    gap: ${({ theme }) => theme.spacing.xs};
    height: auto;
    line-height: 1.4;
  }

  ${({ $active, $color }) =>
    $active
      ? `
          background-color: ${$color};
          border-color: ${$color};
          color: white;
        `
      : `
          background-color: transparent;
          color: #8B7D7B;
          border-color: #D4C4B0;

          &:hover {
            background-color: ${$color}20;
            border-color: ${$color};
            color: ${$color};
          }
        `
  }
`;

const Icon = styled.span`
  font-size: 18px;
  line-height: 1;
  flex-shrink: 0;

  /* Very small screens (320px and below) */
  @media (max-width: 320px) {
    font-size: 16px;
  }

  /* Small screens (321px - 425px) */
  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: 16px;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 18px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 20px;
  }
`;

const ButtonText = styled.span<{ $active?: boolean }>`
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
  flex: 1;

  /* Hide text on mobile for inactive buttons */
  @media (max-width: 425px) {
    ${({ $active }) => !$active && `
      display: none;
    `}

    ${({ $active }) => $active && `
      display: inline-block;
      font-size: 10px;
      font-weight: 600;
      margin-top: 2px;
    `}
  }

  /* Tablet and desktop: always show text */
  @media (min-width: 426px) {
    display: inline-block;
  }
`;

interface NavigationProps {
  currentView: 'timer' | 'tasks' | 'dashboard' | 'community' | 'profile';
  onViewChange?: (view: 'timer' | 'tasks' | 'dashboard' | 'community' | 'profile') => void;
  className?: string;
  style?: React.CSSProperties;
}

export const Navigation: React.FC<NavigationProps> = ({
  currentView,
  onViewChange,
  className,
  style,
}) => {
  const navigate = useNavigate();
  const location = useLocation();

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
    {
      id: 'profile' as const,
      label: 'Profile',
      icon: '👤',
      color: '#C85A5A',
      path: '/profile',
    },
  ];

  const handleNavClick = (item: typeof navItems[0]) => {
    navigate(item.path);
    onViewChange?.(item.id);
  };

  // Determine active view based on current location
  const activeView = navItems.find(item => item.path === location.pathname)?.id || currentView;

  return (
    <NavContainer className={className} style={style}>
      <NavContent>
        {navItems.map((item) => (
          <NavButton
            key={item.id}
            variant={activeView === item.id ? 'primary' : 'secondary'}
            onClick={() => handleNavClick(item)}
            $active={activeView === item.id}
            $color={item.color}
            data-active={activeView === item.id}
          >
            <Icon>{item.icon}</Icon>
            <ButtonText $active={activeView === item.id}>{item.label}</ButtonText>
          </NavButton>
        ))}
      </NavContent>
    </NavContainer>
  );
};

export type { NavigationProps };