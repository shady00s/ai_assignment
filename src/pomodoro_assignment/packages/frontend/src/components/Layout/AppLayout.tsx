import React from 'react';
import styled from 'styled-components';
import { useLocation, Outlet } from 'react-router-dom';
import { useAppSelector } from '../../hooks/redux';
import { authSelectors } from '../../store/slices/authSlice';
import { useGlobalTimer, useThemeToggle } from '../../hooks';
import { EnhancedNavigation } from '../organisms/EnhancedNavigation';

interface AppLayoutProps {
  children: React.ReactNode;
}

const AppContainer = styled.div`
  min-height: 100vh;
  width: 100%;
  overflow-x: hidden;

  /* Default light mode background */
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.neutral[50]} 0%, ${({ theme }) => theme.colors.neutral[100]} 100%);
  color: ${({ theme }) => theme.colors.neutral[800]};

  /* Dark mode styles using global class targeting */
  .dark-mode & {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
    color: #F1F5F9 !important;
  }
`;





const NavigationWrapper = styled.div<{ $showNavigation: boolean }>`
  padding: ${({ theme, $showNavigation }) => $showNavigation ? theme.spacing.mobile.lg : '0'}
    ${({ theme }) => theme.spacing.mobile.md};
  margin-bottom: ${({ theme, $showNavigation }) => $showNavigation ? theme.spacing.mobile.lg : '0'};

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme, $showNavigation }) => $showNavigation ? theme.spacing.tablet.xl : '0'}
      ${({ theme }) => theme.spacing.tablet.md};
    margin-bottom: ${({ theme, $showNavigation }) => $showNavigation ? theme.spacing.tablet.xl : '0'};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme, $showNavigation }) => $showNavigation ? '40px' : '0'} 20px;
    margin-bottom: ${({ theme, $showNavigation }) => $showNavigation ? '32px' : '0'};
  }
`;

const MainContent = styled.main<{ $showNavigation: boolean }>`
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
  padding-bottom: ${({ $showNavigation }) => $showNavigation ? '80px' : '0'};
`;

const AuthContent = styled.div`
  width: 100%;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const AppLayout: React.FC<AppLayoutProps> = ({ children = <Outlet /> }) => {
  const location = useLocation();

  const isAuthenticated = useAppSelector(authSelectors.selectIsAuthenticated);
  const { isDarkMode, toggleTheme } = useThemeToggle();

  // Initialize global timer hook - this ensures timer continues running during navigation
  useGlobalTimer();

  // Determine if we should show the header and navigation
  const isAuthPage = location.pathname === '/auth';
  const isOnboardingPage = location.pathname === '/onboarding';
  const showHeaderAndNavigation = isAuthenticated && !isAuthPage && !isOnboardingPage;

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
        <NavigationWrapper $showNavigation={showHeaderAndNavigation}>
          <EnhancedNavigation
            onMenuToggle={() => console.log('Menu toggle')}
            onNotificationsClick={() => console.log('Notifications clicked')}
            onThemeToggle={toggleTheme}
            isDarkMode={isDarkMode}
          />
        </NavigationWrapper>
      )}

      <MainContent $showNavigation={showHeaderAndNavigation}>
        {children}
      </MainContent>
    </AppContainer>
  );
};