// Redux hooks
export { useAppDispatch, useAppSelector } from './redux';

// Timer hooks
export { useGlobalTimer } from './useGlobalTimer';

// Theme hooks
export { useThemeToggle } from './useThemeToggle';

// Wellness hooks
export { useWellnessData } from './useWellnessData';
export { useWellnessReminders } from './useWellnessReminders';
export { useWellnessGoals } from './useWellnessGoals';
export { useWellnessNotifications,
  createHydrationNotification,
  createMovementNotification,
  createMeditationNotification,
  createMoodCheckInNotification
} from './useWellnessNotifications';

// Other hooks will be added here as we create them
// export { useTimer } from './useTimer';
// export { useLocalStorage } from './useLocalStorage';
// export { useOnlineStatus } from './useOnlineStatus';