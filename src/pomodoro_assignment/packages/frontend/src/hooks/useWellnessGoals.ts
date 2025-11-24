import { useState, useEffect, useCallback } from 'react';
import { useSelector } from 'react-redux';
import {
  useGetWellnessGoalsQuery,
  useCreateWellnessGoalMutation,
  useUpdateWellnessGoalMutation,
  useDeleteWellnessGoalMutation,
} from '@/store/api/wellnessApi';
import { RootState } from '../store';
import { WellnessGoal } from '../types';

export interface WellnessGoalProgress {
  goal: WellnessGoal;
  current: number;
  percentage: number;
  isCompleted: boolean;
  streak?: number;
}

export interface UseWellnessGoalsReturn {
  goals: WellnessGoal[];
  isLoading: boolean;
  isCreating: boolean;
  isUpdating: boolean;
  isDeleting: boolean;
  activeGoals: WellnessGoal[];
  goalsProgress: WellnessGoalProgress[];
  createGoal: (goal: Omit<WellnessGoal, 'id' | 'userId' | 'createdAt' | 'updatedAt'>) => Promise<WellnessGoal>;
  updateGoal: (id: string, updates: Partial<WellnessGoal>) => Promise<WellnessGoal>;
  deleteGoal: (id: string) => Promise<void>;
  getGoalByCategory: (category: WellnessGoal['category']) => WellnessGoal | undefined;
  calculateGoalProgress: (goal: WellnessGoal) => WellnessGoalProgress;
  getCompletedGoals: () => WellnessGoalProgress[];
  getActiveGoals: () => WellnessGoalProgress[];
}

export const useWellnessGoals = (): UseWellnessGoalsReturn => {
  const { data: goals = [], isLoading } = useGetWellnessGoalsQuery();
  const [createGoalMutation, { isLoading: isCreating }] = useCreateWellnessGoalMutation();
  const [updateGoalMutation, { isLoading: isUpdating }] = useUpdateWellnessGoalMutation();
  const [deleteGoalMutation, { isLoading: isDeleting }] = useDeleteWellnessGoalMutation();

  const todayWellness = useSelector((state: RootState) => state.wellnessApi?.queries?.['getTodayWellness(undefined)']?.data);

  // Filter active goals
  const activeGoals = goals.filter(goal => goal.active);

  // Calculate progress for all goals
  const goalsProgress = activeGoals.map(goal => calculateGoalProgress(goal));

  // Create goal
  const createGoal = useCallback(async (
    goalData: Omit<WellnessGoal, 'id' | 'userId' | 'createdAt' | 'updatedAt'>
  ): Promise<WellnessGoal> => {
    try {
      const result = await createGoalMutation(goalData).unwrap();
      return result;
    } catch (error) {
      console.error('Failed to create goal:', error);
      throw error;
    }
  }, [createGoalMutation]);

  // Update goal
  const updateGoal = useCallback(async (
    id: string,
    updates: Partial<WellnessGoal>
  ): Promise<WellnessGoal> => {
    try {
      const result = await updateGoalMutation({ id, updates }).unwrap();
      return result;
    } catch (error) {
      console.error('Failed to update goal:', error);
      throw error;
    }
  }, [updateGoalMutation]);

  // Delete goal
  const deleteGoal = useCallback(async (id: string): Promise<void> => {
    try {
      await deleteGoalMutation({ id }).unwrap();
    } catch (error) {
      console.error('Failed to delete goal:', error);
      throw error;
    }
  }, [deleteGoalMutation]);

  // Get goal by category
  const getGoalByCategory = useCallback((
    category: WellnessGoal['category']
  ): WellnessGoal | undefined => {
    return activeGoals.find(goal => goal.category === category && goal.active);
  }, [activeGoals]);

  // Calculate goal progress
  const calculateGoalProgress = useCallback((goal: WellnessGoal): WellnessGoalProgress => {
    let current = 0;

    switch (goal.category) {
      case 'HYDRATION':
        current = todayWellness?.hydrationGlasses || 0;
        break;
      case 'MOVEMENT':
        // Use movement breaks as the primary metric
        current = todayWellness?.movementBreaks || 0;
        break;
      case 'MEDITATION':
        current = Math.floor((todayWellness?.meditationMinutes || 0) / 15); // Convert minutes to sessions
        break;
      case 'SLEEP':
        current = todayWellness?.sleepHours || 0;
        break;
      default:
        current = 0;
    }

    const percentage = Math.min(100, Math.round((current / goal.targetValue) * 100));
    const isCompleted = current >= goal.targetValue;

    return {
      goal,
      current,
      percentage,
      isCompleted,
      streak: calculateGoalStreak(goal),
    };
  }, [todayWellness]);

  // Get completed goals
  const getCompletedGoals = useCallback((): WellnessGoalProgress[] => {
    return goalsProgress.filter(progress => progress.isCompleted);
  }, [goalsProgress]);

  // Get active goals
  const getActiveGoals = useCallback((): WellnessGoalProgress[] => {
    return goalsProgress.filter(progress => !progress.isCompleted);
  }, [goalsProgress]);

  return {
    goals,
    isLoading,
    isCreating,
    isUpdating,
    isDeleting,
    activeGoals,
    goalsProgress,
    createGoal,
    updateGoal,
    deleteGoal,
    getGoalByCategory,
    calculateGoalProgress,
    getCompletedGoals,
    getActiveGoals,
  };
};

// Helper function to calculate goal streak
const calculateGoalStreak = (goal: WellnessGoal): number => {
  // This would typically involve checking historical data
  // For now, we'll return a placeholder value
  // In a real implementation, you'd query wellness history and calculate consecutive days the goal was met
  return 0;
};

// Utility function to get default goal values
export const getDefaultGoalValues = (): Record<WellnessGoal['category'], { daily: number; weekly: number; monthly: number }> => {
  return {
    HYDRATION: { daily: 8, weekly: 56, monthly: 240 }, // glasses of water
    MOVEMENT: { daily: 5, weekly: 25, monthly: 100 }, // movement breaks
    MEDITATION: { daily: 15, weekly: 105, monthly: 450 }, // minutes
    SLEEP: { daily: 8, weekly: 56, monthly: 240 }, // hours of sleep
  };
};

// Utility function to get goal display info
export const getGoalDisplayInfo = (category: WellnessGoal['category']) => {
  const info = {
    HYDRATION: {
      icon: '💧',
      label: 'Hydration',
      unit: 'glasses',
      color: '#6B8E9F',
      description: 'Daily water intake goal',
    },
    MOVEMENT: {
      icon: '🚶',
      label: 'Movement',
      unit: 'breaks',
      color: '#E67E50',
      description: 'Daily movement breaks goal',
    },
    MEDITATION: {
      icon: '🧘',
      label: 'Meditation',
      unit: 'minutes',
      color: '#7FA870',
      description: 'Daily meditation goal',
    },
    SLEEP: {
      icon: '😴',
      label: 'Sleep',
      unit: 'hours',
      color: '#7A8B7F',
      description: 'Daily sleep goal',
    },
  };

  return info[category];
};

// Utility function to validate goal values
export const validateGoalValue = (category: WellnessGoal['category'], value: number, period: WellnessGoal['period']): boolean => {
  const defaults = getDefaultGoalValues();
  const defaultValue = defaults[category][period];

  // Allow values within a reasonable range of the default
  const minValue = Math.max(1, Math.floor(defaultValue * 0.25));
  const maxValue = Math.ceil(defaultValue * 3);

  return value >= minValue && value <= maxValue;
};

// Utility function to suggest goal improvements
export const getGoalSuggestions = (currentProgress: WellnessGoalProgress[]) => {
  const suggestions: string[] = [];

  currentProgress.forEach((progress) => {
    const { goal, percentage } = progress;
    const displayInfo = getGoalDisplayInfo(goal.category);

    if (percentage === 0) {
      suggestions.push(`Start tracking your ${displayInfo.label.toLowerCase()} to build healthy habits!`);
    } else if (percentage < 50) {
      suggestions.push(`You're ${percentage}% of the way to your ${displayInfo.label.toLowerCase()} goal. Keep going!`);
    } else if (percentage >= 100) {
      suggestions.push(`Great job hitting your ${displayInfo.label.toLowerCase()} goal! Consider increasing it for next ${goal.period.toLowerCase()}.`);
    }
  });

  return suggestions;
};

export default useWellnessGoals;