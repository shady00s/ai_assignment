import React, { useState } from 'react';
import styled from 'styled-components';
import { useLocation, Outlet } from 'react-router-dom';
import { useAppSelector } from '../../hooks/redux';
import { authSelectors } from '../../store/slices/authSlice';
import { Navigation } from '../organisms';
import { useGlobalTimer } from '../../hooks';

interface AppLayoutProps {
  children: React.ReactNode;
}

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.neutral[50]} 0%, ${({ theme }) => theme.colors.neutral[100]} 100%);
  width: 100%;
  overflow-x: hidden;
`;

const AppHeader = styled.header<{ $showHeader: boolean }>`
  text-align: center;
  padding: ${({ theme, $showHeader }) => $showHeader ? theme.spacing.mobile.lg : theme.spacing.mobile.md}
    ${({ theme }) => theme.spacing.mobile.md};
  margin-bottom: ${({ theme, $showHeader }) => $showHeader ? theme.spacing.mobile.lg : '0'};

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme, $showHeader }) => $showHeader ? theme.spacing.tablet.xl : theme.spacing.tablet.md}
      ${({ theme }) => theme.spacing.tablet.md};
    margin-bottom: ${({ theme, $showHeader }) => $showHeader ? theme.spacing.tablet.xl : '0'};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme, $showHeader }) => $showHeader ? '40px' : '20px'} 20px;
    margin-bottom: ${({ theme, $showHeader }) => $showHeader ? '32px' : '0'};
  }
`;

const AppTitle = styled.h1<{ $showHeader: boolean }>`
  font-size: ${({ theme, $showHeader }) => $showHeader ? theme.typography.fontSize.mobile['3xl'] : theme.typography.fontSize.mobile['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.neutral[500]};
  margin: 0 0 ${({ theme }) => theme.spacing.mobile.sm} 0;
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};
  display: ${({ $showHeader }) => $showHeader ? 'block' : 'none'};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme, $showHeader }) => $showHeader ? theme.typography.fontSize.tablet['4xl'] : theme.typography.fontSize.tablet['2xl']};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: ${({ theme, $showHeader }) => $showHeader ? '2.5rem' : '1.5rem'};
  }
`;

const AppSubtitle = styled.p<{ $showHeader: boolean }>`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  color: ${({ theme }) => theme.colors.neutral[400]};
  margin: 0;
  display: ${({ $showHeader }) => $showHeader ? 'block' : 'none'};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 1.1rem;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 1.1rem;
  }
`;

const MainContent = styled.main<{ $showNavigation: boolean }>`
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
  padding-bottom: ${({ theme, $showNavigation }) => $showNavigation ? '80px' : '0'};
`;

const AuthContent = styled.div`
  width: 100%;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const AppLayout: React.FC<AppLayoutProps> = ({ children = <Outlet /> }) => {
  const [currentView, setCurrentView] = useState<'timer' | 'tasks' | 'dashboard' | 'community' | 'profile'>('timer');
  const location = useLocation();

  const isAuthenticated = useAppSelector(authSelectors.selectIsAuthenticated);

  // Initialize global timer hook - this ensures timer continues running during navigation
  useGlobalTimer();

  // Determine if we should show the header and navigation
  const isAuthPage = location.pathname === '/auth';
  const isOnboardingPage = location.pathname === '/onboarding';
  const showHeaderAndNavigation = isAuthenticated && !isAuthPage && !isOnboardingPage;

  // Map current path to view for Navigation component
  React.useEffect(() => {
    const path = location.pathname;
    switch (path) {
      case '/timer':
        setCurrentView('timer');
        break;
      case '/tasks':
        setCurrentView('tasks');
        break;
      case '/dashboard':
        setCurrentView('dashboard');
        break;
      case '/community':
        setCurrentView('community');
        break;
      case '/profile':
        setCurrentView('profile');
        break;
      default:
        setCurrentView('timer');
    }
  }, [location.pathname]);

  if (isAuthPage || isOnboardingPage) {
    return (
      <AppContainer>
        <AuthContent>
          {children}
        </AuthContent>
      </AppContainer>
    );
  }

  return (
    <AppContainer>
      {showHeaderAndNavigation && (
        <AppHeader $showHeader={showHeaderAndNavigation}>
          <AppTitle $showHeader={showHeaderAndNavigation}>🌿 OptoPomodoro</AppTitle>
          <AppSubtitle $showHeader={showHeaderAndNavigation}>
            Find your flow, achieve your goals
          </AppSubtitle>
        </AppHeader>
      )}

      {showHeaderAndNavigation && (
        <Navigation
          currentView={currentView}
          onViewChange={setCurrentView}
        />
      )}

      <MainContent $showNavigation={showHeaderAndNavigation}>
        {children}
      </MainContent>
    </AppContainer>
  );
};