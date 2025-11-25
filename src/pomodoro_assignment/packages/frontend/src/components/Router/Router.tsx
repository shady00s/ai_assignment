import React, { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAppSelector } from '../../hooks/redux';
import { authSelectors } from '../../store/slices/authSlice';
import { AuthScreen, OnboardingScreen, TimerScreen, TaskBoardScreen, DashboardScreen, ProfileScreen } from '../pages';
import { AppLayout } from '../Layout';

interface RouterProps {
  children?: React.ReactNode;
}

// Protected route component for authenticated users
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isAuthenticated = useAppSelector(authSelectors.selectIsAuthenticated);

  if (!isAuthenticated) {
    return <Navigate to="/auth" replace />;
  }

  return <>{children}</>;
};

// Public route component for non-authenticated users only
const PublicRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isAuthenticated = useAppSelector(authSelectors.selectIsAuthenticated);
  const user = useAppSelector(authSelectors.selectUser);

  if (isAuthenticated) {
    // Check if user needs onboarding first
    if (user?.preferences?.workDuration) {
      return <Navigate to="/timer" replace />;
    } else {
      return <Navigate to="/onboarding" replace />;
    }
  }

  return <>{children}</>;
};

// Onboarding route component - only for newly authenticated users
const OnboardingRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isAuthenticated = useAppSelector(authSelectors.selectIsAuthenticated);
  const user = useAppSelector(authSelectors.selectUser);

  if (!isAuthenticated) {
    return <Navigate to="/auth" replace />;
  }

  // Check if user has completed onboarding (has preferences)
  if (user?.preferences?.workDuration) {
    return <Navigate to="/timer" replace />;
  }

  return <>{children}</>;
};

export const Router: React.FC<RouterProps> = ({ children }) => {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/auth" element={
          <PublicRoute>
            <AuthScreen />
          </PublicRoute>
        } />

        <Route path="/onboarding" element={
          <OnboardingRoute>
            <OnboardingScreen />
          </OnboardingRoute>
        } />

        {/* Protected routes */}
        <Route path="/timer" element={
          <ProtectedRoute>
            <AppLayout>
              <TimerScreen />
            </AppLayout>
          </ProtectedRoute>
        } />

        <Route path="/tasks" element={
          <ProtectedRoute>
            <AppLayout>
              <TaskBoardScreen />
            </AppLayout>
          </ProtectedRoute>
        } />

        <Route path="/dashboard" element={
          <ProtectedRoute>
            <AppLayout>
              <DashboardScreen />
            </AppLayout>
          </ProtectedRoute>
        } />

        <Route path="/community" element={
          <ProtectedRoute>
            <AppLayout>
              <div style={{
                textAlign: 'center',
                padding: '60px 20px',
                color: '#8B7D7B'
              }}>
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>👥</div>
                <h2 style={{ color: '#2C3E50', marginBottom: '8px' }}>Community</h2>
                <p>Coming soon - Connect with your team and share achievements</p>
              </div>
            </AppLayout>
          </ProtectedRoute>
        } />

        <Route path="/profile" element={
          <ProtectedRoute>
            <AppLayout>
              <ProfileScreen />
            </AppLayout>
          </ProtectedRoute>
        } />

        {/* Root route redirect */}
        <Route path="/" element={<Navigate to="/timer" replace />} />

        {/* Catch all route */}
        <Route path="*" element={<Navigate to="/timer" replace />} />
      </Routes>
    </BrowserRouter>
  );
};