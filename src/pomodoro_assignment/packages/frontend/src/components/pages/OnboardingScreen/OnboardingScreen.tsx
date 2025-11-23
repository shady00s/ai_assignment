import React, { useState, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useAppDispatch, useAppSelector } from '../../../hooks/redux';
import { updateUser, authSelectors } from '../../../store/slices/authSlice';
import { useUpdateProfileMutation } from '../../../store/api/apiSlice';
import { Button } from '../../atoms/Button';
import { Input } from '../../atoms/Input';
import { User, UserPreferences } from '../../../types';

interface OnboardingScreenProps {
  className?: string;
}

const slideIn = keyframes`
  from { opacity: 0; transform: translateX(30px); }
  to { opacity: 1; transform: translateX(0); }
`;

const fadeInUp = keyframes`
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
`;

const floatGently = keyframes`
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-15px); }
`;

const pulse = keyframes`
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.05); opacity: 0.8; }
`;

const OnboardingContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.neutral[50]} 0%, ${({ theme }) => theme.colors.neutral[100]} 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing.lg};
  position: relative;
  overflow: hidden;

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
  opacity: 0.03;
  font-size: 3rem;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing.xl};
  pointer-events: none;

  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: 2rem;
    gap: ${({ theme }) => theme.spacing.lg};
  }
`;

const ZenSymbol = styled.div<{ $delay?: number; $float?: boolean }>`
  animation: ${({ $float = true }) => $float ? floatGently : 'none'} 4s ease-in-out infinite;
  animation-delay: ${({ $delay = 0 }) => $delay}s;
`;

const OnboardingCard = styled(motion.div)`
  background: white;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  box-shadow: ${({ theme }) => theme.shadows.lg};
  padding: ${({ theme }) => theme.spacing.xl};
  width: 100%;
  max-width: 600px;
  position: relative;
  z-index: 1;
  min-height: 600px;
  display: flex;
  flex-direction: column;

  ${({ theme }) => theme.mediaQueries.mobile} {
    padding: ${({ theme }) => theme.spacing.lg};
    max-width: 100%;
    min-height: 500px;
  }
`;

const ProgressIndicator = styled.div`
  display: flex;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing.sm};
  margin-bottom: ${({ theme }) => theme.spacing.xl};
`;

const ProgressDot = styled.div<{ $active: boolean; $completed: boolean }>`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: ${({ theme, $active, $completed }) =>
    $active
      ? theme.colors.primary.main
      : $completed
      ? theme.colors.success
      : theme.colors.neutral[200]};
  cursor: pointer;
  transition: all ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.easeInOut};

  &:hover {
    transform: scale(1.2);
  }
`;

const StepContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
`;

const StepIcon = styled(motion.div)`
  font-size: 4rem;
  margin-bottom: ${({ theme }) => theme.spacing.xl};
  animation: ${floatGently} 4s ease-in-out infinite;

  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: 3rem;
    margin-bottom: ${({ theme }) => theme.spacing.lg};
  }
`;

const StepTitle = styled.h2`
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};
  font-size: ${({ theme }) => theme.typography.fontSize['3xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.neutral[500]};
  margin: 0 0 ${({ theme }) => theme.spacing.md} 0;
  animation: ${fadeInUp} 0.8s ease-out;

  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  }
`;

const StepDescription = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  color: ${({ theme }) => theme.colors.neutral[400]};
  line-height: ${({ theme }) => theme.typography.lineHeight.relaxed};
  margin: 0 0 ${({ theme }) => theme.spacing.xl} 0;
  max-width: 400px;
  animation: ${fadeInUp} 0.8s ease-out 0.2s both;

  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: ${({ theme }) => theme.typography.fontSize.base};
    max-width: 100%;
  }
`;

const StepForm = styled.form`
  width: 100%;
  max-width: 400px;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.lg};
  animation: ${fadeInUp} 0.8s ease-out 0.4s both;
`;

const PreferencesGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing.md};

  ${({ theme }) => theme.mediaQueries.mobile} {
    grid-template-columns: 1fr;
  }
`;

const PreferenceCard = styled(motion.div)<{ $selected: boolean }>`
  background: ${({ theme, $selected }) =>
    $selected ? theme.colors.primary.main + '15' : theme.colors.neutral[50]};
  border: 2px solid ${({ theme, $selected }) =>
    $selected ? theme.colors.primary.main : theme.colors.neutral[200]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.md};
  cursor: pointer;
  transition: all ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.easeInOut};
  text-align: center;

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.md};
  }

  ${({ theme }) => theme.mediaQueries.mobile} {
    padding: ${({ theme }) => theme.spacing.sm};
  }
`;

const PreferenceIcon = styled.div`
  font-size: 2rem;
  margin-bottom: ${({ theme }) => theme.spacing.sm};

  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: 1.5rem;
  }
`;

const PreferenceTitle = styled.div`
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.neutral[500]};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
`;

const PreferenceDescription = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.neutral[400]};
`;

const RangeContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const RangeHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const RangeLabel = styled.div`
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.neutral[500]};
`;

const RangeValue = styled.div`
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.primary.main};
`;

const RangeSlider = styled.input`
  width: 100%;
  height: 6px;
  background: ${({ theme }) => theme.colors.neutral[200]};
  border-radius: 3px;
  outline: none;
  -webkit-appearance: none;

  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 20px;
    height: 20px;
    background: ${({ theme }) => theme.colors.primary.main};
    border-radius: 50%;
    cursor: pointer;
    box-shadow: ${({ theme }) => theme.shadows.sm};
    transition: all ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.easeInOut};

    &:hover {
      transform: scale(1.1);
    }
  }

  &::-moz-range-thumb {
    width: 20px;
    height: 20px;
    background: ${({ theme }) => theme.colors.primary.main};
    border-radius: 50%;
    cursor: pointer;
    box-shadow: ${({ theme }) => theme.shadows.sm};
    border: none;
    transition: all ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.easeInOut};

    &:hover {
      transform: scale(1.1);
    }
  }
`;

const ActionButtons = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.md};
  margin-top: ${({ theme }) => theme.spacing.xl};

  ${({ theme }) => theme.mediaQueries.mobile} {
    flex-direction: column;
    gap: ${({ theme }) => theme.spacing.sm};
  }
`;

const SkipButton = styled.button`
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.neutral[400]};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  transition: all ${({ theme }) => theme.animation.duration.normal} ${({ theme }) => theme.animation.easing.easeInOut};

  &:hover {
    color: ${({ theme }) => theme.colors.primary.main};
    background: ${({ theme }) => theme.colors.neutral[50]};
  }
`;

interface OnboardingData {
  // Step 1: Focus Preferences
  workDuration: number;
  shortBreakDuration: number;
  longBreakDuration: number;

  // Step 2: Ambient Preferences
  ambientSound: 'forest' | 'ocean' | 'cafe' | 'rain' | 'none';
  soundEnabled: boolean;

  // Step 3: Wellness Preferences
  mindfulnessReminders: boolean;
  hydrationReminders: boolean;
  movementBreaks: boolean;
}

const STEPS = [
  {
    id: 'focus',
    icon: '🎯',
    title: 'Customize Your Focus',
    description: 'Set your ideal work and break durations to match your productivity rhythm',
  },
  {
    id: 'ambient',
    icon: '🎵',
    title: 'Choose Your Atmosphere',
    description: 'Select ambient sounds and settings to create your perfect work environment',
  },
  {
    id: 'wellness',
    icon: '🌿',
    title: 'Balance Your Well-being',
    description: 'Enable wellness reminders to maintain health during focused work sessions',
  },
];

const AMBIENT_SOUNDS = [
  { value: 'forest', icon: '🌲', title: 'Forest', description: 'Peaceful woodland sounds' },
  { value: 'ocean', icon: '🌊', title: 'Ocean', description: 'Calming waves and sea breeze' },
  { value: 'cafe', icon: '☕', title: 'Café', description: 'Ambient coffee shop buzz' },
  { value: 'rain', icon: '🌧️', title: 'Rain', description: 'Soothing rainfall sounds' },
  { value: 'none', icon: '🔇', title: 'Silent', description: 'No background sounds' },
] as const;

export const OnboardingScreen: React.FC<OnboardingScreenProps> = ({ className }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [onboardingData, setOnboardingData] = useState<OnboardingData>({
    workDuration: 25,
    shortBreakDuration: 5,
    longBreakDuration: 15,
    ambientSound: 'forest',
    soundEnabled: true,
    mindfulnessReminders: true,
    hydrationReminders: true,
    movementBreaks: true,
  });

  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const { user, isLoading } = useAppSelector((state) => ({
    user: authSelectors.selectUser(state),
    isLoading: authSelectors.selectIsLoading(state),
  }));

  const [updateProfile, { isLoading: isUpdating }] = useUpdateProfileMutation();

  // Redirect to dashboard if user is not authenticated or doesn't need onboarding
  useEffect(() => {
    if (!user) {
      navigate('/auth');
      return;
    }

    // If user already has preferences set, skip onboarding
    if (user.preferences && user.preferences.workDuration) {
      navigate('/timer');
      return;
    }
  }, [user, navigate]);

  const updateOnboardingData = (field: keyof OnboardingData, value: any) => {
    setOnboardingData(prev => ({ ...prev, [field]: value }));
  };

  const handleStepClick = (stepIndex: number) => {
    if (stepIndex < currentStep) {
      setCurrentStep(stepIndex);
    }
  };

  const handleNext = () => {
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      handleComplete();
    }
  };

  const handleComplete = async () => {
    try {
      const preferences: UserPreferences = {
        workDuration: onboardingData.workDuration,
        shortBreakDuration: onboardingData.shortBreakDuration,
        longBreakDuration: onboardingData.longBreakDuration,
        longBreakInterval: 4,
        autoStartBreaks: false,
        autoStartWork: false,
        soundEnabled: onboardingData.soundEnabled,
        volume: 70,
        ambientSound: onboardingData.ambientSound,
        darkMode: false,
        notifications: {
          achievements: true,
          teamUpdates: true,
          weeklyReports: true,
          deadlineReminders: true,
          wellnessReminders: true,
        },
        wellness: {
          mindfulnessReminders: onboardingData.mindfulnessReminders,
          hydrationReminders: onboardingData.hydrationReminders,
          movementBreaks: onboardingData.movementBreaks,
          eyeRest: true,
          endOfDay: false,
        },
      };

      // First update local Redux state immediately
      dispatch(updateUser({ preferences }));

      // Try to update on backend, but don't fail if it doesn't work
      try {
        await updateProfile({ preferences }).unwrap();
      } catch (backendError) {
        console.warn('Failed to save preferences to backend, but saved locally:', backendError);
      }

      navigate('/timer');
    } catch (error) {
      console.error('Failed to complete onboarding:', error);
    }
  };

  const handleSkip = () => {
    navigate('/timer');
  };

  const renderStepContent = () => {
    switch (STEPS[currentStep].id) {
      case 'focus':
        return (
          <StepForm>
            <RangeContainer>
              <RangeHeader>
                <RangeLabel>Focus Duration</RangeLabel>
                <RangeValue>{onboardingData.workDuration} min</RangeValue>
              </RangeHeader>
              <RangeSlider
                type="range"
                min="15"
                max="60"
                step="5"
                value={onboardingData.workDuration}
                onChange={(e) => updateOnboardingData('workDuration', parseInt(e.target.value))}
              />
            </RangeContainer>

            <RangeContainer>
              <RangeHeader>
                <RangeLabel>Short Break</RangeLabel>
                <RangeValue>{onboardingData.shortBreakDuration} min</RangeValue>
              </RangeHeader>
              <RangeSlider
                type="range"
                min="3"
                max="15"
                step="1"
                value={onboardingData.shortBreakDuration}
                onChange={(e) => updateOnboardingData('shortBreakDuration', parseInt(e.target.value))}
              />
            </RangeContainer>

            <RangeContainer>
              <RangeHeader>
                <RangeLabel>Long Break</RangeLabel>
                <RangeValue>{onboardingData.longBreakDuration} min</RangeValue>
              </RangeHeader>
              <RangeSlider
                type="range"
                min="10"
                max="30"
                step="5"
                value={onboardingData.longBreakDuration}
                onChange={(e) => updateOnboardingData('longBreakDuration', parseInt(e.target.value))}
              />
            </RangeContainer>
          </StepForm>
        );

      case 'ambient':
        return (
          <StepForm>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={onboardingData.soundEnabled}
                  onChange={(e) => updateOnboardingData('soundEnabled', e.target.checked)}
                  style={{ width: '18px', height: '18px' }}
                />
                <span style={{ fontWeight: '500', color: '#2C3E50' }}>
                  Enable ambient sounds
                </span>
              </label>
            </div>

            {onboardingData.soundEnabled && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: '12px' }}>
                {AMBIENT_SOUNDS.map((sound) => (
                  <PreferenceCard
                    key={sound.value}
                    $selected={onboardingData.ambientSound === sound.value}
                    onClick={() => updateOnboardingData('ambientSound', sound.value)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <PreferenceIcon>{sound.icon}</PreferenceIcon>
                    <PreferenceTitle>{sound.title}</PreferenceTitle>
                    <PreferenceDescription>{sound.description}</PreferenceDescription>
                  </PreferenceCard>
                ))}
              </div>
            )}
          </StepForm>
        );

      case 'wellness':
        return (
          <StepForm>
            <PreferencesGrid>
              <PreferenceCard
                $selected={onboardingData.mindfulnessReminders}
                onClick={() => updateOnboardingData('mindfulnessReminders', !onboardingData.mindfulnessReminders)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <PreferenceIcon>🧘</PreferenceIcon>
                <PreferenceTitle>Mindfulness</PreferenceTitle>
                <PreferenceDescription>Take mindful breaks</PreferenceDescription>
              </PreferenceCard>

              <PreferenceCard
                $selected={onboardingData.hydrationReminders}
                onClick={() => updateOnboardingData('hydrationReminders', !onboardingData.hydrationReminders)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <PreferenceIcon>💧</PreferenceIcon>
                <PreferenceTitle>Hydration</PreferenceTitle>
                <PreferenceDescription>Stay hydrated</PreferenceDescription>
              </PreferenceCard>

              <PreferenceCard
                $selected={onboardingData.movementBreaks}
                onClick={() => updateOnboardingData('movementBreaks', !onboardingData.movementBreaks)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <PreferenceIcon>🚶</PreferenceIcon>
                <PreferenceTitle>Movement</PreferenceTitle>
                <PreferenceDescription>Regular movement breaks</PreferenceDescription>
              </PreferenceCard>
            </PreferencesGrid>
          </StepForm>
        );

      default:
        return null;
    }
  };

  return (
    <OnboardingContainer className={className}>
      <ZenBackground>
        {['🌿', '🪨', '💧', '🎋', '🍃', '🪵', '🌸', '🍁', '🎵', '🧘'].map((symbol, index) => (
          <ZenSymbol key={index} $delay={index * 0.3}>
            {symbol}
          </ZenSymbol>
        ))}
      </ZenBackground>

      <OnboardingCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <ProgressIndicator>
          {STEPS.map((_, index) => (
            <ProgressDot
              key={index}
              $active={index === currentStep}
              $completed={index < currentStep}
              onClick={() => handleStepClick(index)}
            />
          ))}
        </ProgressIndicator>

        <StepContent>
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -30 }}
              transition={{ duration: 0.4 }}
              style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1 }}
            >
              <StepIcon>{STEPS[currentStep].icon}</StepIcon>
              <StepTitle>{STEPS[currentStep].title}</StepTitle>
              <StepDescription>{STEPS[currentStep].description}</StepDescription>
              {renderStepContent()}
            </motion.div>
          </AnimatePresence>

          <ActionButtons>
            <SkipButton type="button" onClick={handleSkip}>
              {currentStep === STEPS.length - 1 ? 'Finish Later' : 'Skip for now'}
            </SkipButton>
            <Button
              variant="primary"
              size="large"
              onClick={handleNext}
              loading={isLoading || isUpdating}
              disabled={isLoading || isUpdating}
            >
              {currentStep === STEPS.length - 1 ? 'Complete Setup' : 'Continue'}
            </Button>
          </ActionButtons>
        </StepContent>
      </OnboardingCard>
    </OnboardingContainer>
  );
};

export default OnboardingScreen;