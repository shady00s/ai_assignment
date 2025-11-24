import { useState, useEffect, useCallback, useRef } from "react";
import { useSelector } from "react-redux";
import {
  useCreateWellnessReminderMutation,
  useDeleteWellnessReminderMutation,
  useGetWellnessRemindersQuery,
  useToggleWellnessReminderMutation,
  useUpdateWellnessReminderMutation,
} from "@/store/api/wellnessApi";

import { RootState } from "../store";
import { WellnessReminder } from "../types";

export interface UseWellnessRemindersReturn {
  reminders: WellnessReminder[];
  notificationPermission: NotificationPermission;
  isLoading: boolean;
  isCreating: boolean;
  activeReminders: WellnessReminder[];
  requestNotificationPermission: () => Promise<boolean>;
  createReminder: (
    reminder: Omit<
      WellnessReminder,
      "id" | "userId" | "createdAt" | "updatedAt" | "lastTrigger"
    >
  ) => Promise<WellnessReminder>;
  updateReminder: (
    id: string,
    updates: Partial<WellnessReminder>
  ) => Promise<WellnessReminder>;
  toggleReminder: (id: string, enabled: boolean) => Promise<WellnessReminder>;
  deleteReminder: (id: string) => Promise<void>;
  scheduleReminder: (reminder: WellnessReminder) => void;
  cancelReminder: (id: string) => void;
  testReminder: (type: WellnessReminder["type"]) => void;
}

export const useWellnessReminders = (): UseWellnessRemindersReturn => {
  const [notificationPermission, setNotificationPermission] =
    useState<NotificationPermission>("default");
  const [activeReminders, setActiveReminders] = useState<WellnessReminder[]>(
    []
  );
  const [scheduledIntervals, setScheduledIntervals] = useState<
    Map<string, NodeJS.Timeout>
  >(new Map());

  const { data: reminders = [], isLoading } = useGetWellnessRemindersQuery();
  const [createReminderMutation, { isLoading: isCreating }] =
    useCreateWellnessReminderMutation();
  const [updateReminderMutation] = useUpdateWellnessReminderMutation();
  const [toggleReminderMutation] = useToggleWellnessReminderMutation();
  const [deleteReminderMutation] = useDeleteWellnessReminderMutation();

  // Check notification permission on mount
  useEffect(() => {
    if ("Notification" in window) {
      setNotificationPermission(Notification.permission);
    }
  }, []);

  // Schedule reminders when they change
  useEffect(() => {
    // Clear existing intervals
    scheduledIntervals.forEach((interval) => clearInterval(interval));
    setScheduledIntervals(new Map());

    // Schedule active reminders
    const enabledReminders = reminders.filter((reminder) => reminder.enabled);
    setActiveReminders(enabledReminders);

    enabledReminders.forEach((reminder) => {
      if (shouldScheduleReminder(reminder)) {
        const interval = scheduleReminderInterval(reminder);
        if (interval) {
          setScheduledIntervals((prev) =>
            new Map(prev).set(reminder.id, interval)
          );
        }
      }
    });

    return () => {
      // Cleanup intervals on unmount
      scheduledIntervals.forEach((interval) => clearInterval(interval));
    };
  }, [reminders]);

  // Request notification permission
  const requestNotificationPermission =
    useCallback(async (): Promise<boolean> => {
      if (!("Notification" in window)) {
        console.warn("This browser does not support desktop notifications");
        return false;
      }

      if (Notification.permission === "granted") {
        setNotificationPermission("granted");
        return true;
      }

      if (Notification.permission !== "denied") {
        const permission = await Notification.requestPermission();
        setNotificationPermission(permission);
        return permission === "granted";
      }

      return false;
    }, []);

  // Create reminder
  const createReminder = useCallback(
    async (
      reminderData: Omit<
        WellnessReminder,
        "id" | "userId" | "createdAt" | "updatedAt" | "lastTrigger"
      >
    ): Promise<WellnessReminder> => {
      try {
        const result = await createReminderMutation(reminderData).unwrap();
        return result;
      } catch (error) {
        console.error("Failed to create reminder:", error);
        throw error;
      }
    },
    [createReminderMutation]
  );

  // Update reminder
  const updateReminder = useCallback(
    async (
      id: string,
      updates: Partial<WellnessReminder>
    ): Promise<WellnessReminder> => {
      try {
        const result = await updateReminderMutation({ id, updates }).unwrap();
        return result;
      } catch (error) {
        console.error("Failed to update reminder:", error);
        throw error;
      }
    },
    [updateReminderMutation]
  );

  // Toggle reminder
  const toggleReminder = useCallback(
    async (id: string, enabled: boolean): Promise<WellnessReminder> => {
      try {
        const result = await toggleReminderMutation({ id, enabled }).unwrap();
        return result;
      } catch (error) {
        console.error("Failed to toggle reminder:", error);
        throw error;
      }
    },
    [toggleReminderMutation]
  );

  // Delete reminder
  const deleteReminder = useCallback(
    async (id: string): Promise<void> => {
      try {
        await deleteReminderMutation({ id }).unwrap();

        // Clear scheduled interval if exists
        const interval = scheduledIntervals.get(id);
        if (interval) {
          clearInterval(interval);
          setScheduledIntervals((prev) => {
            const newMap = new Map(prev);
            newMap.delete(id);
            return newMap;
          });
        }
      } catch (error) {
        console.error("Failed to delete reminder:", error);
        throw error;
      }
    },
    [deleteReminderMutation, scheduledIntervals]
  );

  // Schedule reminder logic
  const scheduleReminder = useCallback(
    (reminder: WellnessReminder): void => {
      if (notificationPermission !== "granted") return;

      sendWellnessNotification(reminder.type);
    },
    [notificationPermission]
  );

  // Cancel reminder
  const cancelReminder = useCallback(
    (id: string): void => {
      const interval = scheduledIntervals.get(id);
      if (interval) {
        clearInterval(interval);
        setScheduledIntervals((prev) => {
          const newMap = new Map(prev);
          newMap.delete(id);
          return newMap;
        });
      }
    },
    [scheduledIntervals]
  );

  // Test reminder (for user testing)
  const testReminder = useCallback(
    (type: WellnessReminder["type"]): void => {
      if (notificationPermission === "granted") {
        sendWellnessNotification(type, true);
      }
    },
    [notificationPermission]
  );

  return {
    reminders,
    notificationPermission,
    isLoading,
    isCreating,
    activeReminders,
    requestNotificationPermission,
    createReminder,
    updateReminder,
    toggleReminder,
    deleteReminder,
    scheduleReminder,
    cancelReminder,
    testReminder,
  };
};

// Helper functions
const shouldScheduleReminder = (reminder: WellnessReminder): boolean => {
  if (!reminder.enabled) return false;

  const now = new Date();
  const currentTime = now.getHours() * 60 + now.getMinutes(); // Current time in minutes
  const [startHour, startMin] = reminder.startTime.split(":").map(Number);
  const [endHour, endMin] = reminder.endTime.split(":").map(Number);

  const startTime = startHour * 60 + startMin;
  const endTime = endHour * 60 + endMin;

  // Check if current time is within the reminder's active window
  const isWithinTimeWindow = currentTime >= startTime && currentTime <= endTime;

  // Check if today is a scheduled weekday
  const currentDay = now.getDay(); // 0 = Sunday, 1 = Monday, etc.
  const isScheduledDay = reminder.weekdays.includes(
    currentDay === 0 ? 7 : currentDay
  ); // Convert Sunday to 7

  return isWithinTimeWindow && isScheduledDay;
};

const scheduleReminderInterval = (
  reminder: WellnessReminder
): NodeJS.Timeout | null => {
  if (!shouldScheduleReminder(reminder)) return null;

  return setInterval(
    () => {
      if (shouldScheduleReminder(reminder)) {
        sendWellnessNotification(reminder.type);
      }
    },
    reminder.frequency * 60 * 1000
  ); // Convert frequency from minutes to milliseconds
};

const sendWellnessNotification = (
  type: WellnessReminder["type"],
  isTest = false
): void => {
  if (!("Notification" in window) || Notification.permission !== "granted")
    return;

  const notifications = {
    HYDRATION: {
      title: "💧 Time to Hydrate!",
      body: isTest
        ? "This is a test hydration reminder"
        : "You haven't logged water intake in a while. Time for a glass of water!",
      icon: "/icons/wellness/hydration.png",
      tag: "hydration-reminder",
    },
    MOVEMENT: {
      title: "🚶 Movement Break!",
      body: isTest
        ? "This is a test movement reminder"
        : "You've been focused for a while. Time for a quick movement break!",
      icon: "/icons/wellness/movement.png",
      tag: "movement-reminder",
    },
    POSTURE: {
      title: "🪑 Posture Check",
      body: isTest
        ? "This is a test posture reminder"
        : "Quick check: Are you sitting with good posture?",
      icon: "/icons/wellness/posture.png",
      tag: "posture-check",
      silent: !isTest,
    },
    EYE_REST: {
      title: "👁️ Eye Rest Break",
      body: isTest
        ? "This is a test eye rest reminder"
        : "Time for the 20-20-20 rule: Look at something 20 feet away for 20 seconds!",
      icon: "/icons/wellness/eye-rest.png",
      tag: "eye-rest-reminder",
    },
    MEDITATION: {
      title: "🧘 Mindfulness Moment",
      body: isTest
        ? "This is a test meditation reminder"
        : "Take a moment for deep breathing and mindfulness.",
      icon: "/icons/wellness/meditation.png",
      tag: "meditation-reminder",
    },
  };

  const notification = notifications[type];
  if (!notification) return;

  const notificationInstance = new Notification(notification.title, {
    body: notification.body,
    icon: notification.icon,
    tag: `${notification.tag}${isTest ? "-test" : ""}`,
    requireInteraction: !isTest,
    silent: notification.silent || false,
  });

  // Auto-close non-interactive notifications after 5 seconds
  if (!notificationInstance.requireInteraction) {
    setTimeout(() => {
      notificationInstance.close();
    }, 5000);
  }
};

export default useWellnessReminders;
