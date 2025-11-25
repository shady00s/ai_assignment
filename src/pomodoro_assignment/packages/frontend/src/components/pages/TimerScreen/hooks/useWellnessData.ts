import { useState, useCallback } from 'react';
 
 import { useGetTodayWellnessQuery, useIncrementHydrationMutation, useLogMovementBreakMutation, useUpdateMoodMutation, useLogMeditationMutation } from '@/store/api/apiSlice';
import { WellnessEntry } from '@/types';

interface UseWellnessDataOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface UseWellnessDataReturn {
  wellnessData: WellnessEntry | null;
  isLoading: boolean;
  error: any;
  incrementHydration: (glasses: number) => Promise<void>;
  logMovement: (duration: number, type: string, intensity?: 'LOW' | 'MEDIUM' | 'HIGH') => Promise<void>;
  updateMood: (mood: number, stress?: number, energy?: number) => Promise<void>;
  logMeditation: (minutes: number, type: string, quality: number, notes?: string) => Promise<void>;
  refetch: () => Promise<void>;
}

export const useWellnessData = (options: UseWellnessDataOptions = {}): UseWellnessDataReturn => {
  const { autoRefresh = false, refreshInterval = 30000 } = options;

  const {
    data: wellnessData,
    isLoading,
    error,
    refetch,
  } = useGetTodayWellnessQuery(undefined, {
    refetchOnMountOrArgChange: true,
    refetchOnWindowFocus: autoRefresh,
    pollingInterval: autoRefresh ? refreshInterval : undefined,
  });

  const [incrementHydration] = useIncrementHydrationMutation();
  const [logMovement] = useLogMovementBreakMutation();
  const [updateMood] = useUpdateMoodMutation();
  const [logMeditation] = useLogMeditationMutation();

  const handleIncrementHydration = useCallback(async (glasses: number) => {
    try {
      await incrementHydration({ glasses }).unwrap();
      await refetch();
    } catch (error) {
      console.error('Failed to increment hydration:', error);
    }
  }, [incrementHydration, refetch]);

  const handleLogMovement = useCallback(async (
    duration: number,
    type: string,
    intensity: 'LOW' | 'MEDIUM' | 'HIGH' = 'MEDIUM'
  ) => {
    try {
      await logMovement({
        duration,
        type,
        intensity,
      }).unwrap();
      await refetch();
    } catch (error) {
      console.error('Failed to log movement:', error);
    }
  }, [logMovement, refetch]);

  const handleUpdateMood = useCallback(async (
    mood: number,
    stress?: number,
    energy?: number
  ) => {
    try {
      await updateMood({
        mood,
        stress: stress ?? wellnessData?.stressLevel ?? 3,
        energy: energy ?? wellnessData?.energyLevel ?? 3,
      }).unwrap();
      await refetch();
    } catch (error) {
      console.error('Failed to update mood:', error);
    }
  }, [updateMood, wellnessData, refetch]);

  const handleLogMeditation = useCallback(async (
    minutes: number,
    type: string,
    quality: number,
    notes?: string
  ) => {
    try {
      await logMeditation({
        minutes,
        type,
        quality,
        notes,
      }).unwrap();
      await refetch();
    } catch (error) {
      console.error('Failed to log meditation:', error);
    }
  }, [logMeditation, refetch]);

  return {
    wellnessData,
    isLoading,
    error,
    incrementHydration: handleIncrementHydration,
    logMovement: handleLogMovement,
    updateMood: handleUpdateMood,
    logMeditation: handleLogMeditation,
    refetch,
  };
};