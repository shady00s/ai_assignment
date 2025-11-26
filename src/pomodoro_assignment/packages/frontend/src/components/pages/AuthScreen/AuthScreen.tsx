import React, { useState, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAppDispatch, useAppSelector } from '../../../hooks/redux';
import { loginUser, registerUser, clearAuth, authSelectors } from '../../../store/slices/authSlice';
import { Button } from '../../atoms/Button';
import { Input } from '../../atoms/Input';

interface AuthScreenProps {
  className?: string;
}


const floatGently = keyframes`
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
`;

const AuthContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.neutral[50]} 0%, ${({ theme }) => theme.colors.neutral[100]} 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing.lg};
  position: relative;
  overflow: hidden;

  /* Dark mode styles */
  .dark-mode & {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
  }

  ${({ theme }) => theme.mediaQueries.mobile} {
    padding: ${({ theme }) => theme.spacing.md};
  }
`;

const ZenBackground = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  opacity: 0.05;
  font-size: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing.xl};
  pointer-events: none;

  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: 1.5rem;
    gap: ${({ theme }) => theme.spacing.lg};
  }
`;

const ZenSymbol = styled.div<{ $delay?: number }>`
  animation: ${floatGently} 3s ease-in-out infinite;
  animation-delay: ${({ $delay = 0 }) => $delay}s;
`;

const AuthCard = styled(motion.div)`
  background: white;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  box-shadow: ${({ theme }) => theme.shadows.lg};
  padding: ${({ theme }) => theme.spacing.xl};
  width: 100%;
  max-width: 440px;
  position: relative;
  z-index: 1;

  /* Dark mode styles */
  .dark-mode & {
    background: #1E293B !important;
    color: #F1F5F9 !important;
  }

  ${({ theme }) => theme.mediaQueries.mobile} {
    padding: ${({ theme }) => theme.spacing.lg};
    max-width: 100%;
  }
`;

const AuthHeader = styled.div`
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing.xl};
`;

const Logo = styled.div`
  font-size: 3rem;
  margin-bottom: ${({ theme }) => theme.spacing.md};
  animation: ${floatGently} 4s ease-in-out infinite;

  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: 2.5rem;
  }
`;

const Title = styled.h1`
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};
  font-size: ${({ theme }) => theme.typography.fontSize['3xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.neutral[500]};
  margin: 0 0 ${({ theme }) => theme.spacing.sm} 0;

  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  }
`;

const Subtitle = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  color: ${({ theme }) => theme.colors.neutral[400]};
  margin: 0;
  line-height: ${({ theme }) => theme.typography.lineHeight.relaxed};
`;

const TabContainer = styled.div`
  display: flex;
  background: ${({ theme }) => theme.colors.neutral[50]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.xs};
  margin-bottom: ${({ theme }) => theme.spacing.xl};

  /* Dark mode styles */
  .dark-mode & {
    background: #334155 !important;
  }
`;

const TabButton = styled.button<{ $active: boolean }>`
  flex: 1;
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border: none;
  background: ${({ theme, $active }) =>
    $active ? 'white' : 'transparent'};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme, $active }) =>
    $active ? theme.colors.primary.main : theme.colors.neutral[400]};
  cursor: pointer;
  transition: all ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.easeInOut};
  box-shadow: ${({ theme, $active }) =>
    $active ? theme.shadows.sm : 'none'};

  &:hover {
    color: ${({ theme }) => theme.colors.primary.main};
  }

  /* Dark mode styles */
  .dark-mode & {
    background: ${({ $active }) => $active ? '#1E293B' : 'transparent'} !important;
    color: ${({ $active, theme }) => $active ? theme.colors.primary.main : theme.colors.neutral[400]} !important;
  }

  ${({ theme }) => theme.mediaQueries.mobile} {
    padding: ${({ theme }) => theme.spacing.sm};
    font-size: ${({ theme }) => theme.typography.fontSize.sm};
  }
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
`;

const FormSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
`;

const Divider = styled.div`
  display: flex;
  align-items: center;
  margin: ${({ theme }) => theme.spacing.lg} 0;
  gap: ${({ theme }) => theme.spacing.md};
`;

const DividerLine = styled.div`
  flex: 1;
  height: 1px;
  background: ${({ theme }) => theme.colors.neutral[200]};
`;

const DividerText = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.neutral[400]};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const ErrorMessage = styled(motion.div)`
  background: ${({ theme }) => theme.colors.error}15;
  border: 1px solid ${({ theme }) => theme.colors.error}30;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  color: ${({ theme }) => theme.colors.error};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const SuccessMessage = styled(motion.div)`
  background: ${({ theme }) => theme.colors.success}15;
  border: 1px solid ${({ theme }) => theme.colors.success}30;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  color: ${({ theme }) => theme.colors.success};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const FooterText = styled.p`
  text-align: center;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.neutral[400]};
  margin-top: ${({ theme }) => theme.spacing.lg};
`;

const FooterLink = styled.button`
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.primary.main};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  text-decoration: underline;
  font-size: inherit;

  &:hover {
    color: ${({ theme }) => theme.colors.primary.dark};
  }
`;

const NameContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing.md};

  ${({ theme }) => theme.mediaQueries.mobile} {
    grid-template-columns: 1fr;
    gap: ${({ theme }) => theme.spacing.sm};
  }
`;

interface FormData {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  confirmPassword: string;
}

type AuthMode = 'login' | 'register';

export const AuthScreen: React.FC<AuthScreenProps> = ({ className }) => {
  const [mode, setMode] = useState<AuthMode>('login');
  const [formData, setFormData] = useState<FormData>({
    email: '',
    password: '',
    firstName: '',
    lastName: '',
    confirmPassword: '',
  });
  const [errors, setErrors] = useState<Partial<FormData>>({});

  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const location = useLocation();

  const { isLoading, error, isAuthenticated, user } = useAppSelector((state) => ({
    isLoading: authSelectors.selectIsLoading(state),
    error: authSelectors.selectError(state),
    isAuthenticated: authSelectors.selectIsAuthenticated(state),
    user: authSelectors.selectUser(state),
  }));

  // Note: Navigation is handled by the Router component's PublicRoute and OnboardingRoute guards
  // AuthScreen only handles the login/register form display

  // Note: Removed clearAuth on unmount as it was interfering with navigation after successful registration

  const validateForm = (): boolean => {
    const newErrors: Partial<FormData> = {};

    // Email validation
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email';
    }

    // Password validation
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }

    if (mode === 'register') {
      // Name validation
      if (!formData.firstName) {
        newErrors.firstName = 'First name is required';
      }
      if (!formData.lastName) {
        newErrors.lastName = 'Last name is required';
      }

      // Confirm password validation
      if (!formData.confirmPassword) {
        newErrors.confirmPassword = 'Please confirm your password';
      } else if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = 'Passwords do not match';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleInputChange = (field: keyof FormData) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFormData(prev => ({ ...prev, [field]: event.target.value }));
    // Clear error for this field when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }));
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!validateForm()) {
      return;
    }

    try {
      if (mode === 'login') {
        await dispatch(loginUser({
          email: formData.email,
          password: formData.password,
        })).unwrap();
      } else {
        await dispatch(registerUser({
          email: formData.email,
          password: formData.password,
          firstName: formData.firstName,
          lastName: formData.lastName,
        })).unwrap();
      }
      // Navigation will happen in the useEffect when isAuthenticated changes
    } catch (err) {
      // Error is handled by the Redux slice
    }
  };

  const handleSSO = async () => {
    // Generate unique user data for Optomatica SSO
    const timestamp = Date.now();
    const randomId = Math.floor(Math.random() * 1000);

    const ssoCredentials = {
      email: `user.${timestamp}${randomId}@optomatica.com`,
      password: 'OptomaticaSSO123!', // Default password for SSO users
      firstName: 'Optomatica',
      lastName: `User${randomId}`,
    };

    try {
      await dispatch(registerUser(ssoCredentials)).unwrap();
      // Navigation will happen in the useEffect when isAuthenticated changes
    } catch (err) {
      console.error('SSO registration failed:', err);
      // If registration fails (maybe user exists), try login
      try {
        await dispatch(loginUser({
          email: ssoCredentials.email,
          password: ssoCredentials.password,
        })).unwrap();
      } catch (loginErr) {
        console.error('SSO login also failed:', loginErr);
      }
    }
  };

  const switchMode = () => {
    setMode(prev => prev === 'login' ? 'register' : 'login');
    setFormData({
      email: '',
      password: '',
      firstName: '',
      lastName: '',
      confirmPassword: '',
    });
    setErrors({});
  };

  return (
    <AuthContainer className={className}>
      <ZenBackground>
        {['🌿', '🪨', '💧', '🎋', '🍃', '🪵', '🌸', '🍁'].map((symbol, index) => (
          <ZenSymbol key={index} $delay={index * 0.2}>
            {symbol}
          </ZenSymbol>
        ))}
      </ZenBackground>

      <AuthCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <AuthHeader>
          <Logo>🌿</Logo>
          <Title>OptoPomodoro</Title>
          <Subtitle>
            {mode === 'login'
              ? 'Welcome back to your mindful workspace'
              : 'Begin your journey to focused productivity'
            }
          </Subtitle>
        </AuthHeader>

        <TabContainer>
          <TabButton
            type="button"
            $active={mode === 'login'}
            onClick={() => setMode('login')}
          >
            Sign In
          </TabButton>
          <TabButton
            type="button"
            $active={mode === 'register'}
            onClick={() => setMode('register')}
          >
            Sign Up
          </TabButton>
        </TabContainer>

        <AnimatePresence mode="wait">
          {error && (
            <ErrorMessage
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              {error}
            </ErrorMessage>
          )}
        </AnimatePresence>

        <Form onSubmit={handleSubmit}>
          <FormSection>
            {mode === 'register' && (
              <NameContainer>
                <Input
                  label="First Name"
                  value={formData.firstName}
                  onChange={handleInputChange('firstName')}
                  error={!!errors.firstName}
                  helperText={errors.firstName}
                  required
                  fullWidth
                />
                <Input
                  label="Last Name"
                  value={formData.lastName}
                  onChange={handleInputChange('lastName')}
                  error={!!errors.lastName}
                  helperText={errors.lastName}
                  required
                  fullWidth
                />
              </NameContainer>
            )}

            <Input
              type="email"
              label="Email Address"
              value={formData.email}
              onChange={handleInputChange('email')}
              error={!!errors.email}
              helperText={errors.email}
              placeholder="you@example.com"
              autoComplete={mode === 'login' ? 'email' : 'new-email'}
              required
              fullWidth
            />

            <Input
              type="password"
              label="Password"
              value={formData.password}
              onChange={handleInputChange('password')}
              error={!!errors.password}
              helperText={errors.password || (mode === 'register' ? 'Must be at least 8 characters' : '')}
              placeholder="••••••••"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              required
              fullWidth
            />

            {mode === 'register' && (
              <Input
                type="password"
                label="Confirm Password"
                value={formData.confirmPassword}
                onChange={handleInputChange('confirmPassword')}
                error={!!errors.confirmPassword}
                helperText={errors.confirmPassword}
                placeholder="••••••••"
                autoComplete="new-password"
                required
                fullWidth
              />
            )}
          </FormSection>

          <Button
            type="submit"
            variant="primary"
            size="large"
            loading={isLoading}
            disabled={isLoading}
            fullWidth
          >
            {isLoading
              ? 'Please wait...'
              : (mode === 'login' ? 'Sign In' : 'Create Account')
            }
          </Button>
        </Form>

        <Divider>
          <DividerLine />
          <DividerText>or continue with</DividerText>
          <DividerLine />
        </Divider>

        <Button
          variant="secondary"
          size="large"
          icon="🏢"
          fullWidth
          onClick={handleSSO}
          loading={isLoading}
          disabled={isLoading}
        >
          {isLoading ? 'Connecting...' : 'Optomatica SSO'}
        </Button>

        <FooterText>
          {mode === 'login' ? (
            <>
              Don't have an account?{' '}
              <FooterLink type="button" onClick={switchMode}>
                Sign up
              </FooterLink>
            </>
          ) : (
            <>
              Already have an account?{' '}
              <FooterLink type="button" onClick={switchMode}>
                Sign in
              </FooterLink>
            </>
          )}
        </FooterText>
      </AuthCard>
    </AuthContainer>
  );
};

export default AuthScreen;