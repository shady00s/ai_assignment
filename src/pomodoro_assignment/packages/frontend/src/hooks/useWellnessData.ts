import { useState, useEffect, useCallback } from 'react';
import {
  useGetTodayWellnessQuery,
  useGetWellnessHistoryQuery,
  useGetWellnessAnalyticsQuery,
  useIncrementHydrationMutation,
  useLogMovementBreakMutation,
  useUpdateMoodMutation,
  useLogMeditationMutation,
  useGetWellnessRecommendationsQuery,
  useGetWellnessGoalsQuery,
  useGetWellnessRemindersQuery,
} from '../store/api/wellnessApi';
import { WellnessEntry, Recommendation } from '../types';

// Utility function to get date X days ago
const getDateDaysAgo = (days: number): string => {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return date.toISOString().split('T')[0];
};

// Utility function to get today's date
const getTodayDate = (): string => {
  return new Date().toISOString().split('T')[0];
};

export const useWellnessData = () => {
  // Base wellness data queries
  const {
    data: todayWellness,
    isLoading: todayLoading,
    error: todayError,
    refetch: refetchToday,
  } = useGetTodayWellnessQuery();

  const {
    data: wellnessHistory,
    isLoading: historyLoading,
    error: historyError,
  } = useGetWellnessHistoryQuery({
    startDate: getDateDaysAgo(30),
    endDate: new Date().toISOString(),
  });

  const {
    data: analytics,
    isLoading: analyticsLoading,
    error: analyticsError,
  } = useGetWellnessAnalyticsQuery({ days: 30 });

  const {
    data: recommendations,
    isLoading: recommendationsLoading,
  } = useGetWellnessRecommendationsQuery();

  // Mutations
  const [incrementHydration, { isLoading: hydrationLoading }] = useIncrementHydrationMutation();
  const [logMovement, { isLoading: movementLoading }] = useLogMovementBreakMutation();
  const [updateMood, { isLoading: moodLoading }] = useUpdateMoodMutation();
  const [logMeditation, { isLoading: meditationLoading }] = useLogMeditationMutation();

  // Computed state
  const isLoading = todayLoading || historyLoading || analyticsLoading || recommendationsLoading;
  const hasError = todayError || historyError || analyticsError;
  const isMutating = hydrationLoading || movementLoading || moodLoading || meditationLoading;

  // Wellness actions with optimistic updates
  const actions = {
    hydrate: async (glasses: number = 1) => {
      try {
        const result = await incrementHydration({ glasses }).unwrap();
        return result;
      } catch (error) {
        console.error('Failed to update hydration:', error);
        throw error;
      }
    },

    logMovement: async (duration: number, type: string = 'BREAK', intensity?: 'LOW' | 'MEDIUM' | 'HIGH') => {
      try {
        const result = await logMovement({ duration, type, intensity }).unwrap();
        return result;
      } catch (error) {
        console.error('Failed to log movement:', error);
        throw error;
      }
    },

    updateMood: async (mood: number, stress: number, energy: number) => {
      try {
        const result = await updateMood({ mood, stress, energy }).unwrap();
        return result;
      } catch (error) {
        console.error('Failed to update mood:', error);
        throw error;
      }
    },

    logMeditation: async (minutes: number, type: string = 'MINDFULNESS', quality: number = 3) => {
      try {
        const result = await logMeditation({ minutes, type, quality }).unwrap();
        return result;
      } catch (error) {
        console.error('Failed to log meditation:', error);
        throw error;
      }
    },

    refreshData: async () => {
      await refetchToday();
    },
  };

  // Computed wellness metrics
  const metrics = {
    hydration: {
      current: todayWellness?.hydrationGlasses || 0,
      goal: todayWellness?.hydrationGoal || 8,
      percentage: Math.min(100, Math.round(((todayWellness?.hydrationGlasses || 0) / (todayWellness?.hydrationGoal || 8)) * 100)),
      streak: calculateHydrationStreak(wellnessHistory || []),
      totalMl: (todayWellness?.hydrationGlasses || 0) * 250, // Assuming 250ml per glass
    },

    movement: {
      currentBreaks: todayWellness?.movementBreaks || 0,
      currentMinutes: todayWellness?.movementMinutes || 0,
      goalBreaks: 5,
      goalMinutes: 30,
      breakPercentage: Math.min(100, Math.round(((todayWellness?.movementBreaks || 0) / 5) * 100)),
      minutePercentage: Math.min(100, Math.round(((todayWellness?.movementMinutes || 0) / 30) * 100)),
      stepsCount: todayWellness?.stepsCount || 0,
    },

    mood: {
      current: todayWellness?.moodRating || 3,
      stress: todayWellness?.stressLevel || 3,
      energy: todayWellness?.energyLevel || 3,
      lastCheckIn: todayWellness?.updatedAt || '',
      trend: calculateMoodTrend(wellnessHistory || []),
    },

    meditation: {
      currentMinutes: todayWellness?.meditationMinutes || 0,
      goalMinutes: 15,
      sessions: todayWellness?.mindfulnessSessions || 0,
      percentage: Math.min(100, Math.round(((todayWellness?.meditationMinutes || 0) / 15) * 100)),
      averageSession: wellnessHistory?.length > 0
        ? Math.round((wellnessHistory.reduce((sum, entry) => sum + entry.meditationMinutes, 0) / wellnessHistory.length))
        : 0,
    },

    wellness: {
      score: calculateWellnessScore(todayWellness),
      trend: analytics?.hydration?.trend || 'STABLE',
    },
  };

  return {
    // Data
    todayWellness,
    wellnessHistory,
    analytics,
    recommendations,

    // Loading states
    isLoading,
    hasError,
    isMutating,

    // Actions
    actions,

    // Computed metrics
    metrics,

    // Convenience getters
    isWellnessDataAvailable: !!todayWellness,
    lastUpdated: todayWellness?.updatedAt,
    date: getTodayDate(),
  };
};

// Helper functions
const calculateHydrationStreak = (history: WellnessEntry[]): number => {
  if (!history.length) return 0;

  let streak = 0;
  const sortedHistory = history.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  for (const entry of sortedHistory) {
    if (entry.hydrationGlasses >= entry.hydrationGoal) {
      streak++;
    } else {
      break;
    }
  }

  return streak;
};

const calculateMoodTrend = (history: WellnessEntry[]): 'IMPROVING' | 'DECLINING' | 'STABLE' => {
  if (!history.length || history.length < 3) return 'STABLE';

  const recent = history.slice(-7).map(entry => entry.moodRating);
  const earlier = history.slice(-14, -7).map(entry => entry.moodRating);

  if (earlier.length === 0) return 'STABLE';

  const recentAvg = recent.reduce((sum, mood) => sum + mood, 0) / recent.length;
  const earlierAvg = earlier.reduce((sum, mood) => sum + mood, 0) / earlier.length;

  if (recentAvg > earlierAvg + 0.2) return 'IMPROVING';
  if (recentAvg < earlierAvg - 0.2) return 'DECLINING';
  return 'STABLE';
};

const calculateWellnessScore = (wellness?: WellnessEntry | null): number => {
  if (!wellness) return 0;

  const hydrationScore = Math.min(100, (wellness.hydrationGlasses / wellness.hydrationGoal) * 100);
  const movementScore = Math.min(100, (wellness.movementMinutes / 30) * 100);
  const moodScore = (wellness.moodRating / 5) * 100;
  const meditationScore = Math.min(100, (wellness.meditationMinutes / 15) * 100);
  const stressScore = ((6 - wellness.stressLevel) / 5) * 100; // Invert stress (lower is better)

  return Math.round((hydrationScore + movementScore + moodScore + meditationScore + stressScore) / 5);
};