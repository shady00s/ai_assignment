import { useState, useEffect, useCallback, useRef } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';

export interface WellnessNotification {
  id: string;
  type: 'hydration' | 'movement' | 'posture' | 'eye-rest' | 'meditation' | 'achievement';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  priority: 'low' | 'medium' | 'high';
  autoClose?: boolean;
  duration?: number;
  action?: {
    label: string;
    callback: () => void;
  };
}

export interface UseWellnessNotificationsReturn {
  notifications: WellnessNotification[];
  hasUnreadNotifications: boolean;
  unreadCount: number;
  isPermissionGranted: boolean;
  isSupported: boolean;
  requestPermission: () => Promise<boolean>;
  addNotification: (notification: Omit<WellnessNotification, 'id' | 'timestamp' | 'read'>) => void;
  removeNotification: (id: string) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  clearAll: () => void;
  sendBrowserNotification: (title: string, options?: NotificationOptions) => void;
  scheduleNotification: (type: WellnessNotification['type'], delay: number) => void;
}

export const useWellnessNotifications = (): UseWellnessNotificationsReturn => {
  const [notifications, setNotifications] = useState<WellnessNotification[]>([]);
  const [isPermissionGranted, setIsPermissionGranted] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const scheduledTimeouts = useRef<Map<string, NodeJS.Timeout>>(new Map());

  const user = useSelector((state: RootState) => state.auth.user);

  // Check browser notification support
  useEffect(() => {
    setIsSupported('Notification' in window);
    setIsPermissionGranted('Notification' in window && Notification.permission === 'granted');
  }, []);

  // Auto-remove notifications with duration
  useEffect(() => {
    const intervals: NodeJS.Timeout[] = [];

    notifications.forEach((notification) => {
      if (notification.autoClose && notification.duration) {
        const interval = setTimeout(() => {
          removeNotification(notification.id);
        }, notification.duration);
        intervals.push(interval);
      }
    });

    return () => {
      intervals.forEach(clearInterval);
    };
  }, [notifications]);

  // Request notification permission
  const requestPermission = useCallback(async (): Promise<boolean> => {
    if (!isSupported) {
      console.warn('Notifications are not supported in this browser');
      return false;
    }

    if (isPermissionGranted) {
      return true;
    }

    try {
      const permission = await Notification.requestPermission();
      setIsPermissionGranted(permission === 'granted');
      return permission === 'granted';
    } catch (error) {
      console.error('Failed to request notification permission:', error);
      return false;
    }
  }, [isSupported, isPermissionGranted]);

  // Add notification
  const addNotification = useCallback((
    notification: Omit<WellnessNotification, 'id' | 'timestamp' | 'read'>
  ) => {
    const newNotification: WellnessNotification = {
      ...notification,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      read: false,
    };

    setNotifications(prev => [newNotification, ...prev]);

    // Send browser notification if permission is granted
    if (isPermissionGranted && notification.priority !== 'low') {
      sendBrowserNotification(notification.title, {
        body: notification.message,
        icon: `/icons/wellness/${notification.type}.png`,
        tag: notification.type,
        requireInteractive: notification.priority === 'high',
      });
    }

    return newNotification;
  }, [isPermissionGranted]);

  // Remove notification
  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));

    // Cancel any scheduled timeout for this notification
    const timeout = scheduledTimeouts.current.get(id);
    if (timeout) {
      clearTimeout(timeout);
      scheduledTimeouts.current.delete(id);
    }
  }, []);

  // Mark as read
  const markAsRead = useCallback((id: string) => {
    setNotifications(prev =>
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    );
  }, []);

  // Mark all as read
  const markAllAsRead = useCallback(() => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  }, []);

  // Clear all notifications
  const clearAll = useCallback(() => {
    setNotifications([]);
    // Clear all scheduled timeouts
    scheduledTimeouts.current.forEach(timeout => clearTimeout(timeout));
    scheduledTimeouts.current.clear();
  }, []);

  // Send browser notification
  const sendBrowserNotification = useCallback((
    title: string,
    options: NotificationOptions = {}
  ) => {
    if (!isSupported || !isPermissionGranted) {
      return;
    }

    const notification = new Notification(title, {
      icon: '/icons/wellness/default.png',
      badge: '/icons/wellness/badge.png',
      ...options,
    });

    // Auto-close after 5 seconds unless requireInteraction is true
    if (!options.requireInteraction) {
      setTimeout(() => {
        notification.close();
      }, 5000);
    }

    return notification;
  }, [isSupported, isPermissionGranted]);

  // Schedule notification
  const scheduleNotification = useCallback((
    type: WellnessNotification['type'],
    delay: number
  ) => {
    const notificationMessages = {
      hydration: {
        title: '💧 Hydration Reminder',
        message: 'Time to drink some water! Stay hydrated throughout the day.',
      },
      movement: {
        title: '🚶 Movement Break',
        message: 'You\'ve been sitting for a while. Time for a quick movement break!',
      },
      posture: {
        title: '🪑 Posture Check',
        message: 'Check your posture! Sit up straight and relax your shoulders.',
      },
      'eye-rest': {
        title: '👁️ Eye Rest Break',
        message: 'Follow the 20-20-20 rule: Look at something 20 feet away for 20 seconds.',
      },
      meditation: {
        title: '🧘 Mindfulness Moment',
        message: 'Take a moment for deep breathing and mental clarity.',
      },
      achievement: {
        title: '🎉 Achievement Unlocked!',
        message: 'Congratulations! You\'ve reached a wellness milestone.',
      },
    };

    const message = notificationMessages[type];
    if (!message) return;

    const timeoutId = setTimeout(() => {
      addNotification({
        type,
        ...message,
        priority: 'medium',
        autoClose: true,
        duration: 5000,
      });
    }, delay);

    const id = `${type}-${Date.now()}`;
    scheduledTimeouts.current.set(id, timeoutId);

    return id;
  }, [addNotification]);

  // Computed values
  const hasUnreadNotifications = notifications.some(n => !n.read);
  const unreadCount = notifications.filter(n => !n.read).length;

  return {
    notifications,
    hasUnreadNotifications,
    unreadCount,
    isPermissionGranted,
    isSupported,
    requestPermission,
    addNotification,
    removeNotification,
    markAsRead,
    markAllAsRead,
    clearAll,
    sendBrowserNotification,
    scheduleNotification,
  };
};

// Utility functions for common wellness notifications
export const createHydrationNotification = (glassesConsumed: number, dailyGoal: number) => ({
  type: 'hydration' as const,
  title: '💧 Great job staying hydrated!',
  message: `You've had ${glassesConsumed} glasses of water today. ${dailyGoal - glassesConsumed} more to reach your goal!`,
  priority: 'medium' as const,
  autoClose: true,
  duration: 3000,
});

export const createMovementNotification = (breaksTaken: number, dailyGoal: number) => ({
  type: 'movement' as const,
  title: '🚶 Movement Break Complete!',
  message: `Great job! You've taken ${breaksTaken} movement breaks today. ${dailyGoal - breaksTaken} more to reach your goal!`,
  priority: 'medium' as const,
  autoClose: true,
  duration: 3000,
});

export const createMeditationNotification = (minutes: number) => ({
  type: 'meditation' as const,
  title: '🧘 Meditation Session Complete',
  message: `Well done! You completed a ${minute} minute meditation session for mental clarity.`,
  priority: 'low' as const,
  autoClose: true,
  duration: 3000,
});

export const createMoodCheckInNotification = () => ({
  type: 'meditation' as const,
  title: '😊 How are you feeling?',
  message: 'Take a moment to check in with your mood and track your emotional wellness.',
  priority: 'low' as const,
  autoClose: false,
  action: {
    label: 'Check In Now',
    callback: () => {
      // This would typically open the mood tracker modal
      console.log('Open mood tracker');
    },
  },
});

export default useWellnessNotifications;