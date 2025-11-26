import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Notification } from '../../types';

// Helper function to create notification with required fields
const createNotification = (baseNotification: Partial<Omit<Notification, 'id' | 'timestamp' | 'read' | 'priority' | 'userId'>>, priority: Notification['priority'] = 'MEDIUM'): Notification => ({
  id: `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  timestamp: new Date().toISOString(),
  read: false,
  priority,
  userId: 'current-user', // This would normally come from auth state
  ...baseNotification,
} as Notification);

interface UIState {
  sidebarOpen: boolean;
  theme: 'light' | 'dark' | 'auto';
  currentView: string;
  loading: boolean;
  error: string | null;
  notifications: Notification[];
  focused: boolean;
  soundEnabled: boolean;
  volume: number;
  ambientSound: 'forest' | 'ocean' | 'cafe' | 'rain' | 'none';
  screenSize: 'mobile' | 'tablet' | 'desktop';
  online: boolean;
  keyboardShortcuts: boolean;
  animations: boolean;
  compactMode: boolean;
  showZenGarden: boolean;
  zenGardenState: {
    stones: number;
    raked: boolean;
    waterFlow: boolean;
    bambooGrowth: number;
  };
}

const getInitialTheme = (): 'light' | 'dark' | 'auto' => {
  // Check localStorage first
  const savedTheme = localStorage.getItem('theme-mode');
  if (savedTheme === 'light' || savedTheme === 'dark' || savedTheme === 'auto') {
    return savedTheme;
  }

  // Fall back to system preference
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

const initialState: UIState = {
  sidebarOpen: true,
  theme: getInitialTheme(),
  currentView: 'timer',
  loading: false,
  error: null,
  notifications: [],
  focused: false,
  soundEnabled: true,
  volume: 70,
  ambientSound: 'none',
  screenSize: 'desktop',
  online: navigator.onLine,
  keyboardShortcuts: true,
  animations: true,
  compactMode: false,
  showZenGarden: true,
  zenGardenState: {
    stones: 0,
    raked: false,
    waterFlow: false,
    bambooGrowth: 0,
  },
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.sidebarOpen = action.payload;
    },
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setTheme: (state, action: PayloadAction<'light' | 'dark' | 'auto'>) => {
      state.theme = action.payload;
      // Save to localStorage for persistence
      localStorage.setItem('theme-mode', action.payload);
    },
  toggleTheme: (state) => {
    // Simple toggle between light and dark (ignoring auto)
    const newTheme = state.theme === 'dark' ? 'light' : 'dark';
    state.theme = newTheme;
    localStorage.setItem('theme-mode', newTheme);
  },
    setCurrentView: (state, action: PayloadAction<string>) => {
      state.currentView = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp' | 'read'>>) => {
      const notification: Notification = {
        id: `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toISOString(),
        read: false,
        ...action.payload,
      };
      state.notifications.unshift(notification);

      // Keep only last 50 notifications
      if (state.notifications.length > 50) {
        state.notifications = state.notifications.slice(0, 50);
      }
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    markNotificationAsRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification) {
        notification.read = true;
      }
    },
    markAllNotificationsAsRead: (state) => {
      state.notifications.forEach(n => {
        n.read = true;
      });
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    setFocused: (state, action: PayloadAction<boolean>) => {
      state.focused = action.payload;
    },
    setSoundEnabled: (state, action: PayloadAction<boolean>) => {
      state.soundEnabled = action.payload;
    },
    setVolume: (state, action: PayloadAction<number>) => {
      state.volume = Math.max(0, Math.min(100, action.payload));
    },
    setAmbientSound: (state, action: PayloadAction<'forest' | 'ocean' | 'cafe' | 'rain' | 'none'>) => {
      state.ambientSound = action.payload;
    },
    setScreenSize: (state, action: PayloadAction<'mobile' | 'tablet' | 'desktop'>) => {
      state.screenSize = action.payload;

      // Auto-adjust sidebar for mobile
      if (action.payload === 'mobile') {
        state.sidebarOpen = false;
      }
    },
    setOnline: (state, action: PayloadAction<boolean>) => {
      state.online = action.payload;

      if (!action.payload) {
        // Show offline notification
        const offlineNotification = createNotification({
          type: 'WARNING',
          title: 'Offline Mode',
          message: 'You are currently offline. Some features may be limited.',
        }, 'HIGH');
        state.notifications.unshift(offlineNotification);
      } else {
        // Show online notification
        const onlineNotification = createNotification({
          type: 'SUCCESS',
          title: 'Back Online',
          message: 'Your connection has been restored.',
        }, 'LOW');
        state.notifications.unshift(onlineNotification);
      }
    },
    setKeyboardShortcuts: (state, action: PayloadAction<boolean>) => {
      state.keyboardShortcuts = action.payload;
    },
    setAnimations: (state, action: PayloadAction<boolean>) => {
      state.animations = action.payload;
    },
    setCompactMode: (state, action: PayloadAction<boolean>) => {
      state.compactMode = action.payload;
    },
    setShowZenGarden: (state, action: PayloadAction<boolean>) => {
      state.showZenGarden = action.payload;
    },
    updateZenGardenState: (state, action: PayloadAction<Partial<UIState['zenGardenState']>>) => {
      state.zenGardenState = { ...state.zenGardenState, ...action.payload };
    },
    addZenStone: (state) => {
      state.zenGardenState.stones += 1;
      state.zenGardenState.raked = false; // Reset rake when adding stone
    },
    rakeZenGarden: (state) => {
      state.zenGardenState.raked = true;
    },
    growZenBamboo: (state) => {
      state.zenGardenState.bambooGrowth = Math.min(100, state.zenGardenState.bambooGrowth + 10);
    },
    startWaterFlow: (state) => {
      state.zenGardenState.waterFlow = true;
    },
    stopWaterFlow: (state) => {
      state.zenGardenState.waterFlow = false;
    },
    resetZenGarden: (state) => {
      state.zenGardenState = {
        stones: 0,
        raked: false,
        waterFlow: false,
        bambooGrowth: 0,
      };
    },
    showSuccessNotification: (state, action: PayloadAction<{ title: string; message?: string }>) => {
      const notification = createNotification({
        type: 'SUCCESS',
        title: action.payload.title,
        message: action.payload.message || '',
      });
      state.notifications.unshift(notification);
    },
    showErrorNotification: (state, action: PayloadAction<{ title: string; message?: string }>) => {
      const notification = createNotification({
        type: 'ERROR',
        title: action.payload.title,
        message: action.payload.message || '',
      }, 'HIGH');
      state.notifications.unshift(notification);
    },
    showWarningNotification: (state, action: PayloadAction<{ title: string; message?: string }>) => {
      const notification = createNotification({
        type: 'WARNING',
        title: action.payload.title,
        message: action.payload.message || '',
      });
      state.notifications.unshift(notification);
    },
    showInfoNotification: (state, action: PayloadAction<{ title: string; message?: string }>) => {
      const notification = createNotification({
        type: 'INFO',
        title: action.payload.title,
        message: action.payload.message || '',
      }, 'LOW');
      state.notifications.unshift(notification);
    },
  },
});

export const {
  setSidebarOpen,
  toggleSidebar,
  setTheme,
  toggleTheme,
  setCurrentView,
  setLoading,
  setError,
  clearError,
  addNotification,
  removeNotification,
  markNotificationAsRead,
  markAllNotificationsAsRead,
  clearNotifications,
  setFocused,
  setSoundEnabled,
  setVolume,
  setAmbientSound,
  setScreenSize,
  setOnline,
  setKeyboardShortcuts,
  setAnimations,
  setCompactMode,
  setShowZenGarden,
  updateZenGardenState,
  addZenStone,
  rakeZenGarden,
  growZenBamboo,
  startWaterFlow,
  stopWaterFlow,
  resetZenGarden,
  showSuccessNotification,
  showErrorNotification,
  showWarningNotification,
  showInfoNotification,
} = uiSlice.actions;

// Selectors
export const uiSelectors = {
  selectSidebarOpen: (state: { ui: UIState }) => state.ui.sidebarOpen,
  selectTheme: (state: { ui: UIState }) => state.ui.theme,
  selectCurrentView: (state: { ui: UIState }) => state.ui.currentView,
  selectLoading: (state: { ui: UIState }) => state.ui.loading,
  selectError: (state: { ui: UIState }) => state.ui.error,
  selectNotifications: (state: { ui: UIState }) => state.ui.notifications,
  selectUnreadNotifications: (state: { ui: UIState }) =>
    state.ui.notifications.filter(n => !n.read),
  selectNotificationCount: (state: { ui: UIState }) =>
    state.ui.notifications.filter(n => !n.read).length,
  selectFocused: (state: { ui: UIState }) => state.ui.focused,
  selectSoundEnabled: (state: { ui: UIState }) => state.ui.soundEnabled,
  selectVolume: (state: { ui: UIState }) => state.ui.volume,
  selectAmbientSound: (state: { ui: UIState }) => state.ui.ambientSound,
  selectScreenSize: (state: { ui: UIState }) => state.ui.screenSize,
  selectOnline: (state: { ui: UIState }) => state.ui.online,
  selectKeyboardShortcuts: (state: { ui: UIState }) => state.ui.keyboardShortcuts,
  selectAnimations: (state: { ui: UIState }) => state.ui.animations,
  selectCompactMode: (state: { ui: UIState }) => state.ui.compactMode,
  selectShowZenGarden: (state: { ui: UIState }) => state.ui.showZenGarden,
  selectZenGardenState: (state: { ui: UIState }) => state.ui.zenGardenState,
  selectIsMobile: (state: { ui: UIState }) => state.ui.screenSize === 'mobile',
  selectIsTablet: (state: { ui: UIState }) => state.ui.screenSize === 'tablet',
  selectIsDesktop: (state: { ui: UIState }) => state.ui.screenSize === 'desktop',
  selectIsDarkMode: (state: { ui: UIState }) =>
    state.ui.theme === 'dark' || (state.ui.theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches),
};

export { uiSlice };
export type { UIState };