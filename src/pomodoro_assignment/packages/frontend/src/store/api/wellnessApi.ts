import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import  { RootState } from '../index';
import {
  WellnessEntry,
  WellnessReminder,
  WellnessGoal,
  DetailedWellnessAnalytics,
  Recommendation,
  IncrementHydrationRequest,
  LogMovementRequest,
  UpdateMoodRequest,
  LogMeditationRequest,
} from '../../types';

// Define tag types for cache invalidation
export const wellnessTagTypes = [
  'WellnessEntry',
  'WellnessReminder',
  'WellnessGoal',
  'WellnessAnalytics',
  'WellnessRecommendation',
] as const;

export const wellnessApi = createApi({
  reducerPath: 'wellnessApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/wellness',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token;
      if (token) {
        headers.set('authorization', `Bearer ${token}`);
      }
      return headers;
    },
  }),
  tagTypes: wellnessTagTypes,
  endpoints: (builder) => ({
    // Wellness Entries
    getTodayWellness: builder.query<WellnessEntry, void>({
      query: () => 'today',
      providesTags: ['WellnessEntry'],
    }),

    getWellnessHistory: builder.query<WellnessEntry[], { startDate: string; endDate: string }>({
      query: ({ startDate, endDate }) => `history?startDate=${startDate}&endDate=${endDate}`,
      providesTags: ['WellnessEntry'],
    }),

    getWellnessEntryByDate: builder.query<WellnessEntry, { date: string }>({
      query: ({ date }) => `entry/${date}`,
      providesTags: (result, error, { date }) => [{ type: 'WellnessEntry', id: date }],
    }),

    createWellnessEntry: builder.mutation<WellnessEntry, Partial<WellnessEntry>>({
      query: (entry) => ({
        url: 'entry',
        method: 'POST',
        body: entry,
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    updateWellnessEntry: builder.mutation<WellnessEntry, { date: string; updates: Partial<WellnessEntry> }>({
      query: ({ date, updates }) => ({
        url: `entry/${date}`,
        method: 'PUT',
        body: updates,
      }),
      invalidatesTags: (result, error, { date }) => [
        { type: 'WellnessEntry', id: date },
        'WellnessAnalytics',
      ],
    }),

    deleteWellnessEntry: builder.mutation<void, { date: string }>({
      query: ({ date }) => ({
        url: `entry/${date}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, { date }) => [
        { type: 'WellnessEntry', id: date },
        'WellnessAnalytics',
      ],
    }),

    // Hydration endpoints
    incrementHydration: builder.mutation<WellnessEntry, IncrementHydrationRequest>({
      query: ({ glasses }) => ({
        url: 'hydration/increment',
        method: 'POST',
        body: { glasses },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    setHydrationGoal: builder.mutation<WellnessEntry, { goal: number }>({
      query: ({ goal }) => ({
        url: 'hydration/goal',
        method: 'POST',
        body: { goal },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessGoal'],
    }),

    // Movement endpoints
    logMovementBreak: builder.mutation<WellnessEntry, LogMovementRequest>({
      query: ({ duration, type, intensity }) => ({
        url: 'movement/log',
        method: 'POST',
        body: { duration, type, intensity },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    logSteps: builder.mutation<WellnessEntry, { steps: number; date?: string }>({
      query: ({ steps, date }) => ({
        url: 'movement/steps',
        method: 'POST',
        body: { steps, date },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    setMovementGoal: builder.mutation<WellnessEntry, { dailyBreaks: number; dailyMinutes: number }>({
      query: ({ dailyBreaks, dailyMinutes }) => ({
        url: 'movement/goal',
        method: 'POST',
        body: { dailyBreaks, dailyMinutes },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessGoal'],
    }),

    // Mood endpoints
    updateMood: builder.mutation<WellnessEntry, UpdateMoodRequest>({
      query: ({ mood, stress, energy }) => ({
        url: 'mood/update',
        method: 'POST',
        body: { mood, stress, energy },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    logSleep: builder.mutation<WellnessEntry, { hours: number; quality: number; date?: string }>({
      query: ({ hours, quality, date }) => ({
        url: 'mood/sleep',
        method: 'POST',
        body: { hours, quality, date },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    // Meditation endpoints
    logMeditation: builder.mutation<WellnessEntry, LogMeditationRequest>({
      query: ({ minutes, type, quality, notes }) => ({
        url: 'meditation/log',
        method: 'POST',
        body: { minutes, type, quality, notes },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    completeMeditationSession: builder.mutation<WellnessEntry, { sessionId: string; quality: number; notes?: string }>({
      query: ({ sessionId, quality, notes }) => ({
        url: 'meditation/complete',
        method: 'POST',
        body: { sessionId, quality, notes },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    // Breathing exercises
    logBreathingExercise: builder.mutation<WellnessEntry, { duration: number; type: string; rounds?: number }>({
      query: ({ duration, type, rounds }) => ({
        url: 'meditation/breathing',
        method: 'POST',
        body: { duration, type, rounds },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    // Posture and eye rest
    logPostureCheck: builder.mutation<WellnessEntry, { completed: boolean }>({
      query: ({ completed }) => ({
        url: 'posture/check',
        method: 'POST',
        body: { completed },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    logEyeRestBreak: builder.mutation<WellnessEntry, { duration: number; completed: boolean }>({
      query: ({ duration, completed }) => ({
        url: 'eye-rest/break',
        method: 'POST',
        body: { duration, completed },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    // Reminders endpoints
    getWellnessReminders: builder.query<WellnessReminder[], void>({
      query: () => 'reminders',
      providesTags: ['WellnessReminder'],
    }),

    createWellnessReminder: builder.mutation<WellnessReminder, Omit<WellnessReminder, 'id' | 'userId' | 'createdAt' | 'updatedAt' | 'lastTrigger'>>({
      query: (reminder) => ({
        url: 'reminders',
        method: 'POST',
        body: reminder,
      }),
      invalidatesTags: ['WellnessReminder'],
    }),

    updateWellnessReminder: builder.mutation<WellnessReminder, { id: string; updates: Partial<WellnessReminder> }>({
      query: ({ id, updates }) => ({
        url: `reminders/${id}`,
        method: 'PUT',
        body: updates,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'WellnessReminder', id }],
    }),

    deleteWellnessReminder: builder.mutation<void, { id: string }>({
      query: ({ id }) => ({
        url: `reminders/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'WellnessReminder', id }],
    }),

    toggleWellnessReminder: builder.mutation<WellnessReminder, { id: string; enabled: boolean }>({
      query: ({ id, enabled }) => ({
        url: `reminders/${id}/toggle`,
        method: 'POST',
        body: { enabled },
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'WellnessReminder', id }],
    }),

    // Goals endpoints
    getWellnessGoals: builder.query<WellnessGoal[], void>({
      query: () => 'goals',
      providesTags: ['WellnessGoal'],
    }),

    createWellnessGoal: builder.mutation<WellnessGoal, Omit<WellnessGoal, 'id' | 'userId' | 'createdAt' | 'updatedAt'>>({
      query: (goal) => ({
        url: 'goals',
        method: 'POST',
        body: goal,
      }),
      invalidatesTags: ['WellnessGoal'],
    }),

    updateWellnessGoal: builder.mutation<WellnessGoal, { id: string; updates: Partial<WellnessGoal> }>({
      query: ({ id, updates }) => ({
        url: `goals/${id}`,
        method: 'PUT',
        body: updates,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'WellnessGoal', id }],
    }),

    deleteWellnessGoal: builder.mutation<void, { id: string }>({
      query: ({ id }) => ({
        url: `goals/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'WellnessGoal', id }],
    }),

    // Analytics endpoints
    getWellnessAnalytics: builder.query<DetailedWellnessAnalytics, { days?: number; category?: string }>({
      query: ({ days = 30, category }) => {
        const params = new URLSearchParams();
        params.set('days', days.toString());
        if (category) params.set('category', category);
        return `analytics?${params.toString()}`;
      },
      providesTags: ['WellnessAnalytics'],
    }),

    getWellnessTrends: builder.query<any, { period: 'week' | 'month' | 'year' }>({
      query: ({ period }) => `analytics/trends?period=${period}`,
      providesTags: ['WellnessAnalytics'],
    }),

    getWellnessSummary: builder.query<any, { period: 'day' | 'week' | 'month' }>({
      query: ({ period }) => `analytics/summary?period=${period}`,
      providesTags: ['WellnessAnalytics'],
    }),

    // Recommendations endpoint
    getWellnessRecommendations: builder.query<Recommendation[], { limit?: number }>({
      query: ({ limit = 10 }) => `recommendations${limit ? `?limit=${limit}` : ''}`,
      providesTags: ['WellnessRecommendation'],
    }),

    acknowledgeRecommendation: builder.mutation<Recommendation, { id: string; acknowledged: boolean }>({
      query: ({ id, acknowledged }) => ({
        url: `recommendations/${id}/acknowledge`,
        method: 'POST',
        body: { acknowledged },
      }),
      invalidatesTags: ['WellnessRecommendation'],
    }),

    // Wellness scores and achievements
    getWellnessScore: builder.query<{ score: number; breakdown: any; trend: string }, void>({
      query: () => 'score',
      providesTags: ['WellnessAnalytics'],
    }),

    getWellnessAchievements: builder.query<any[], void>({
      query: () => 'achievements',
      providesTags: ['WellnessAnalytics'],
    }),

    // Quick actions for common operations
    quickLogWater: builder.mutation<WellnessEntry, { glasses: number }>({
      query: ({ glasses }) => ({
        url: 'quick/water',
        method: 'POST',
        body: { glasses },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    quickLogMovement: builder.mutation<WellnessEntry, { minutes: number }>({
      query: ({ minutes }) => ({
        url: 'quick/movement',
        method: 'POST',
        body: { minutes },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    quickLogMood: builder.mutation<WellnessEntry, { mood: number; energy: number; stress: number }>({
      query: ({ mood, energy, stress }) => ({
        url: 'quick/mood',
        method: 'POST',
        body: { mood, energy, stress },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    quickLogMeditation: builder.mutation<WellnessEntry, { minutes: number }>({
      query: ({ minutes }) => ({
        url: 'quick/meditation',
        method: 'POST',
        body: { minutes },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),
  }),
});

// Export hooks for usage in components
export const {
  // Wellness Entries
  useGetTodayWellnessQuery,
  useGetWellnessHistoryQuery,
  useGetWellnessEntryByDateQuery,
  useCreateWellnessEntryMutation,
  useUpdateWellnessEntryMutation,
  useDeleteWellnessEntryMutation,

  // Hydration
  useIncrementHydrationMutation,
  useSetHydrationGoalMutation,

  // Movement
  useLogMovementBreakMutation,
  useLogStepsMutation,
  useSetMovementGoalMutation,

  // Mood
  useUpdateMoodMutation,
  useLogSleepMutation,

  // Meditation
  useLogMeditationMutation,
  useCompleteMeditationSessionMutation,
  useLogBreathingExerciseMutation,

  // Posture and eye rest
  useLogPostureCheckMutation,
  useLogEyeRestBreakMutation,

  // Reminders
  useGetWellnessRemindersQuery,
  useCreateWellnessReminderMutation,
  useUpdateWellnessReminderMutation,
  useDeleteWellnessReminderMutation,
  useToggleWellnessReminderMutation,

  // Goals
  useGetWellnessGoalsQuery,
  useCreateWellnessGoalMutation,
  useUpdateWellnessGoalMutation,
  useDeleteWellnessGoalMutation,

  // Analytics
  useGetWellnessAnalyticsQuery,
  useGetWellnessTrendsQuery,
  useGetWellnessSummaryQuery,

  // Recommendations
  useGetWellnessRecommendationsQuery,
  useAcknowledgeRecommendationMutation,

  // Wellness scores and achievements
  useGetWellnessScoreQuery,
  useGetWellnessAchievementsQuery,

  // Quick actions
  useQuickLogWaterMutation,
  useQuickLogMovementMutation,
  useQuickLogMoodMutation,
  useQuickLogMeditationMutation,
} = wellnessApi;