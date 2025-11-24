import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import   { RootState } from '../index';
import {
  User,
  Task,
  Session,
  Team,
  Achievement,
  UserAchievement,
  Challenge,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  CreateTaskRequest,
  UpdateTaskRequest,
  CreateSessionRequest,
  FocusAnalytics,
  WellnessAnalytics,
  TeamAnalytics,
  Notification,
  TaskFilters,
  TaskSort,
  SessionFilters,
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
import { WellnessTrend } from '@/types/wellness.types';

// Define tag types for cache invalidation
export const tagTypes = [
  'User',
  'Task',
  'Session',
  'Team',
  'Achievement',
  'UserAchievement',
  'Challenge',
  'Analytics',
  'Notification',
  'WellnessEntry',
  'WellnessReminder',
  'WellnessGoal',
  'WellnessAnalytics',
  'WellnessRecommendation',
] as const;

export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token;
      if (token) {
        headers.set('authorization', `Bearer ${token}`);
      }
      return headers;
    },
  }),
  tagTypes,
  endpoints: (builder) => ({
    // Auth endpoints
    login: builder.mutation<LoginResponse, LoginRequest>({
      query: (credentials) => ({
        url: 'auth/login',
        method: 'POST',
        body: credentials,
      }),
    }),

    register: builder.mutation<LoginResponse, RegisterRequest>({
      query: (userData) => ({
        url: 'auth/register',
        method: 'POST',
        body: userData,
      }),
    }),

    refreshToken: builder.mutation<{ token: string; refreshToken: string }, void>({
      query: () => ({
        url: 'auth/refresh',
        method: 'POST',
      }),
    }),

    logout: builder.mutation<void, void>({
      query: () => ({
        url: 'auth/logout',
        method: 'POST',
      }),
    }),

    // User endpoints
    getProfile: builder.query<User, void>({
      query: () => 'users/profile',
      providesTags: ['User'],
    }),

    updateProfile: builder.mutation<User, Partial<User>>({
      query: (userData) => ({
        url: 'users/profile',
        method: 'PUT',
        body: userData,
      }),
      invalidatesTags: ['User'],
    }),

    updatePreferences: builder.mutation<User, Partial<User['preferences']>>({
      query: (preferences) => ({
        url: 'users/preferences',
        method: 'PUT',
        body: preferences,
      }),
      invalidatesTags: ['User'],
    }),

    // Task endpoints
    getTasks: builder.query<Task[], { filters?: TaskFilters; sort?: TaskSort }>({
      query: ({ filters, sort }) => {
        const params = new URLSearchParams();

        if (filters) {
          Object.entries(filters).forEach(([key, value]) => {
            if (value) {
              if (Array.isArray(value)) {
                value.forEach(v => params.append(key, v));
              } else {
                params.append(key, value.toString());
              }
            }
          });
        }

        if (sort) {
          params.set('sortBy', sort.field);
          params.set('sortOrder', sort.direction);
        }

        return `tasks?${params.toString()}`;
      },
      providesTags: ['Task'],
    }),

    getTaskById: builder.query<Task, string>({
      query: (id) => `tasks/${id}`,
      providesTags: (result, error, id) => [{ type: 'Task', id }],
    }),

    createTask: builder.mutation<Task, CreateTaskRequest>({
      query: (taskData) => ({
        url: 'tasks',
        method: 'POST',
        body: taskData,
      }),
      invalidatesTags: ['Task'],
    }),

    updateTask: builder.mutation<Task, { id: string; updates: UpdateTaskRequest }>({
      query: ({ id, updates }) => ({
        url: `tasks/${id}`,
        method: 'PUT',
        body: updates,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'Task', id }],
    }),

    deleteTask: builder.mutation<void, string>({
      query: (id) => ({
        url: `tasks/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [{ type: 'Task', id }],
    }),

    // Session endpoints
    getSessions: builder.query<Session[], { filters?: SessionFilters; limit?: number }>({
      query: ({ filters, limit }) => {
        const params = new URLSearchParams();

        if (filters) {
          Object.entries(filters).forEach(([key, value]) => {
            if (value) {
              if (Array.isArray(value)) {
                value.forEach(v => params.append(key, v));
              } else {
                params.append(key, value.toString());
              }
            }
          });
        }

        if (limit) {
          params.set('limit', limit.toString());
        }

        return `sessions?${params.toString()}`;
      },
      providesTags: ['Session'],
    }),

    getSessionById: builder.query<Session, string>({
      query: (id) => `sessions/${id}`,
      providesTags: (result, error, id) => [{ type: 'Session', id }],
    }),

    createSession: builder.mutation<Session, CreateSessionRequest>({
      query: (sessionData) => ({
        url: 'sessions',
        method: 'POST',
        body: sessionData,
      }),
      invalidatesTags: ['Session', 'Analytics'],
    }),

    startSession: builder.mutation<Session, string>({
      query: (id) => ({
        url: `sessions/${id}/start`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [{ type: 'Session', id }],
    }),

    pauseSession: builder.mutation<Session, string>({
      query: (id) => ({
        url: `sessions/${id}/pause`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [{ type: 'Session', id }],
    }),

    completeSession: builder.mutation<Session, { id: string; quality?: number; notes?: string }>({
      query: ({ id, quality, notes }) => ({
        url: `sessions/${id}/complete`,
        method: 'POST',
        body: { quality, notes },
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'Session', id },
        'Analytics',
        'User',
        'Achievement',
        'UserAchievement',
      ],
    }),

    // Team endpoints
    getTeams: builder.query<Team[], void>({
      query: () => 'teams',
      providesTags: ['Team'],
    }),

    getTeamById: builder.query<Team, string>({
      query: (id) => `teams/${id}`,
      providesTags: (result, error, id) => [{ type: 'Team', id }],
    }),

    createTeam: builder.mutation<Team, { name: string; description?: string }>({
      query: (teamData) => ({
        url: 'teams',
        method: 'POST',
        body: teamData,
      }),
      invalidatesTags: ['Team'],
    }),

    joinTeam: builder.mutation<Team, { teamId: string; inviteCode?: string }>({
      query: ({ teamId, inviteCode }) => ({
        url: `teams/${teamId}/join`,
        method: 'POST',
        body: { inviteCode },
      }),
      invalidatesTags: ['Team'],
    }),

    leaveTeam: builder.mutation<void, string>({
      query: (teamId) => ({
        url: `teams/${teamId}/leave`,
        method: 'POST',
      }),
      invalidatesTags: ['Team'],
    }),

    // Achievement endpoints
    getAchievements: builder.query<Achievement[], void>({
      query: () => 'achievements',
      providesTags: ['Achievement'],
    }),

    getUserAchievements: builder.query<UserAchievement[], string>({
      query: (userId) => `users/${userId}/achievements`,
      providesTags: ['UserAchievement'],
    }),

    unlockAchievement: builder.mutation<UserAchievement, { achievementId: string; progress?: number }>({
      query: ({ achievementId, progress }) => ({
        url: `achievements/${achievementId}/unlock`,
        method: 'POST',
        body: { progress },
      }),
      invalidatesTags: ['UserAchievement', 'User'],
    }),

    // Challenge endpoints
    getChallenges: builder.query<Challenge[], { teamId?: string; active?: boolean }>({
      query: ({ teamId, active }) => {
        const params = new URLSearchParams();
        if (teamId) params.set('teamId', teamId);
        if (active !== undefined) params.set('active', active.toString());
        return `challenges?${params.toString()}`;
      },
      providesTags: ['Challenge'],
    }),

    getChallengeById: builder.query<Challenge, string>({
      query: (id) => `challenges/${id}`,
      providesTags: (result, error, id) => [{ type: 'Challenge', id }],
    }),

    createChallenge: builder.mutation<Challenge, Omit<Challenge, 'id' | 'participants' | 'currentValue' | 'createdAt' | 'createdBy'>>({
      query: (challengeData) => ({
        url: 'challenges',
        method: 'POST',
        body: challengeData,
      }),
      invalidatesTags: ['Challenge'],
    }),

    joinChallenge: builder.mutation<Challenge, string>({
      query: (challengeId) => ({
        url: `challenges/${challengeId}/join`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [{ type: 'Challenge', id }],
    }),

    updateChallengeProgress: builder.mutation<Challenge, { challengeId: string; increment: number }>({
      query: ({ challengeId, increment }) => ({
        url: `challenges/${challengeId}/progress`,
        method: 'POST',
        body: { increment },
      }),
      invalidatesTags: (result, error, { challengeId }) => [{ type: 'Challenge', id: challengeId }],
    }),

    // Analytics endpoints
    getFocusAnalytics: builder.query<FocusAnalytics, { startDate?: string; endDate?: string }>({
      query: ({ startDate, endDate }) => {
        const params = new URLSearchParams();
        if (startDate) params.set('startDate', startDate);
        if (endDate) params.set('endDate', endDate);
        return `analytics/focus?${params.toString()}`;
      },
      providesTags: ['Analytics'],
    }),

    getWellnessAnalytics: builder.query<WellnessAnalytics, { startDate?: string; endDate?: string }>({
      query: ({ startDate, endDate }) => {
        const params = new URLSearchParams();
        if (startDate) params.set('startDate', startDate);
        if (endDate) params.set('endDate', endDate);
        return `analytics/wellness?${params.toString()}`;
      },
      providesTags: ['Analytics'],
    }),

    getTeamAnalytics: builder.query<TeamAnalytics, { teamId: string; startDate?: string; endDate?: string }>({
      query: ({ teamId, startDate, endDate }) => {
        const params = new URLSearchParams();
        if (startDate) params.set('startDate', startDate);
        if (endDate) params.set('endDate', endDate);
        return `analytics/teams/${teamId}?${params.toString()}`;
      },
      providesTags: ['Analytics'],
    }),

    // Notification endpoints
    getNotifications: builder.query<Notification[], { unread?: boolean; limit?: number }>({
      query: ({ unread, limit }) => {
        const params = new URLSearchParams();
        if (unread !== undefined) params.set('unread', unread.toString());
        if (limit) params.set('limit', limit.toString());
        return `notifications?${params.toString()}`;
      },
      providesTags: ['Notification'],
    }),

    markNotificationAsRead: builder.mutation<Notification, string>({
      query: (notificationId) => ({
        url: `notifications/${notificationId}/read`,
        method: 'POST',
      }),
      invalidatesTags: ['Notification'],
    }),

    markAllNotificationsAsRead: builder.mutation<Notification[], void>({
      query: () => ({
        url: 'notifications/read-all',
        method: 'POST',
      }),
      invalidatesTags: ['Notification'],
    }),

    // Wellness endpoints
    getTodayWellness: builder.query<WellnessEntry | null, void>({
      query: () => 'wellness/today',
      providesTags: ['WellnessEntry'],
    }),

    getWellnessHistory: builder.query<WellnessEntry[], { startDate?: string; endDate?: string; days?: number; page?: number; limit?: number }>({
      query: ({ startDate, endDate, days, page, limit }) => {
        const params = new URLSearchParams();
        if (startDate) params.set('startDate', startDate);
        if (endDate) params.set('endDate', endDate);
        if (days) params.set('days', days.toString());
        if (page) params.set('page', page.toString());
        if (limit) params.set('limit', limit.toString());
        return `wellness/history?${params.toString()}`;
      },
      providesTags: ['WellnessEntry'],
    }),

    getWellnessGoals: builder.query<WellnessGoal[], void>({
      query: () => 'wellness/goals',
      providesTags: ['WellnessGoal'],
    }),

    createWellnessGoal: builder.mutation<WellnessGoal, Omit<WellnessGoal, 'id' | 'userId' | 'createdAt' | 'updatedAt' | 'currentProgress' | 'progressPercentage'>>({
      query: (goalData) => ({
        url: 'wellness/goals',
        method: 'POST',
        body: goalData,
      }),
      invalidatesTags: ['WellnessGoal', 'WellnessAnalytics'],
    }),

    updateWellnessGoal: builder.mutation<WellnessGoal, { id: string; updates: Partial<WellnessGoal> }>({
      query: ({ id, updates }) => ({
        url: `wellness/goals/${id}`,
        method: 'PUT',
        body: updates,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'WellnessGoal', id }],
    }),

    deleteWellnessGoal: builder.mutation<void, string>({
      query: (id) => ({
        url: `wellness/goals/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [{ type: 'WellnessGoal', id }],
    }),

    getWellnessReminders: builder.query<WellnessReminder[], void>({
      query: () => 'wellness/reminders',
      providesTags: ['WellnessReminder'],
    }),

    createWellnessReminder: builder.mutation<WellnessReminder, Omit<WellnessReminder, 'id' | 'userId' | 'createdAt' | 'updatedAt' | 'lastTrigger'>>({
      query: (reminderData) => ({
        url: 'wellness/reminders',
        method: 'POST',
        body: reminderData,
      }),
      invalidatesTags: ['WellnessReminder'],
    }),

    updateWellnessReminder: builder.mutation<WellnessReminder, { id: string; updates: Partial<WellnessReminder> }>({
      query: ({ id, updates }) => ({
        url: `wellness/reminders/${id}`,
        method: 'PUT',
        body: updates,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'WellnessReminder', id }],
    }),

    deleteWellnessReminder: builder.mutation<void, string>({
      query: (id) => ({
        url: `wellness/reminders/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [{ type: 'WellnessReminder', id }],
    }),

    createOrUpdateWellnessEntry: builder.mutation<WellnessEntry, CreateWellnessEntryRequest>({
      query: (entryData) => ({
        url: 'wellness/entry',
        method: 'POST',
        body: entryData,
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    updateWellnessEntryByDate: builder.mutation<WellnessEntry, { date: string; updates: Partial<CreateWellnessEntryRequest> }>({
      query: ({ date, updates }) => ({
        url: `wellness/entry/${date}`,
        method: 'PUT',
        body: updates,
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    deleteWellnessEntryByDate: builder.mutation<void, string>({
      query: (date) => ({
        url: `wellness/entry/${date}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    getWellnessSummary: builder.query<any, void>({
      query: () => 'wellness/summary',
      providesTags: ['WellnessEntry', 'WellnessGoal'],
    }),

    getWellnessAnalyticsSummary: builder.query<DetailedWellnessAnalytics, { days?: number; startDate?: string; endDate?: string; includeRecommendations?: boolean; includeTrends?: boolean }>({
      query: ({ days, startDate, endDate, includeRecommendations, includeTrends }) => {
        const params = new URLSearchParams();
        if (days) params.set('days', days.toString());
        if (startDate) params.set('startDate', startDate);
        if (endDate) params.set('endDate', endDate);
        if (includeRecommendations !== undefined) params.set('includeRecommendations', includeRecommendations.toString());
        if (includeTrends !== undefined) params.set('includeTrends', includeTrends.toString());
        return `wellness/analytics/summary?${params.toString()}`;
      },
      providesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    getWellnessTrends: builder.query<WellnessTrend[], { days?: number }>({
      query: ({ days }) => {
        const params = new URLSearchParams();
        if (days) params.set('days', days.toString());
        return `wellness/analytics/trends?${params.toString()}`;
      },
      providesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    getWellnessRecommendations: builder.query<WellnessRecommendation[], { days?: number }>({
      query: ({ days }) => {
        const params = new URLSearchParams();
        if (days) params.set('days', days.toString());
        return `wellness/analytics/recommendations?${params.toString()}`;
      },
      providesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    incrementHydration: builder.mutation<WellnessEntry, IncrementHydrationRequest>({
      query: ({ glasses }) => ({
        url: 'wellness/hydration/increment',
        method: 'POST',
        body: { glasses },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    logMovementBreak: builder.mutation<WellnessEntry, LogMovementRequest>({
      query: ({ duration, type, intensity }) => ({
        url: 'wellness/movement/log',
        method: 'POST',
        body: { duration, type, intensity },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    updateMood: builder.mutation<WellnessEntry, UpdateMoodRequest>({
      query: ({ mood, stress, energy }) => ({
        url: 'wellness/mood/update',
        method: 'POST',
        body: { mood, stress, energy },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),

    logMeditation: builder.mutation<WellnessEntry, LogMeditationRequest>({
      query: ({ minutes, type, quality, notes }) => ({
        url: 'wellness/meditation/log',
        method: 'POST',
        body: { minutes, type, quality, notes },
      }),
      invalidatesTags: ['WellnessEntry', 'WellnessAnalytics'],
    }),
  }),
});

// Export hooks for usage in components
export const {
  useLoginMutation,
  useRegisterMutation,
  useRefreshTokenMutation,
  useLogoutMutation,
  useGetProfileQuery,
  useUpdateProfileMutation,
  useUpdatePreferencesMutation,
  useGetTasksQuery,
  useGetTaskByIdQuery,
  useCreateTaskMutation,
  useUpdateTaskMutation,
  useDeleteTaskMutation,
  useGetSessionsQuery,
  useGetSessionByIdQuery,
  useCreateSessionMutation,
  useStartSessionMutation,
  usePauseSessionMutation,
  useCompleteSessionMutation,
  useGetTeamsQuery,
  useGetTeamByIdQuery,
  useCreateTeamMutation,
  useJoinTeamMutation,
  useLeaveTeamMutation,
  useGetAchievementsQuery,
  useGetUserAchievementsQuery,
  useUnlockAchievementMutation,
  useGetChallengesQuery,
  useGetChallengeByIdQuery,
  useCreateChallengeMutation,
  useJoinChallengeMutation,
  useUpdateChallengeProgressMutation,
  useGetFocusAnalyticsQuery,
  useGetWellnessAnalyticsQuery,
  useGetTeamAnalyticsQuery,
  useGetNotificationsQuery,
  useMarkNotificationAsReadMutation,
  useMarkAllNotificationsAsReadMutation,
  useGetTodayWellnessQuery,
  useGetWellnessHistoryQuery,
  useGetWellnessGoalsQuery,
  useCreateWellnessGoalMutation,
  useUpdateWellnessGoalMutation,
  useDeleteWellnessGoalMutation,
  useGetWellnessRemindersQuery,
  useCreateWellnessReminderMutation,
  useUpdateWellnessReminderMutation,
  useDeleteWellnessReminderMutation,
  useCreateOrUpdateWellnessEntryMutation,
  useUpdateWellnessEntryByDateMutation,
  useDeleteWellnessEntryByDateMutation,
  useGetWellnessSummaryQuery,
  useGetWellnessAnalyticsSummaryQuery,
  useGetWellnessTrendsQuery,
  useGetWellnessRecommendationsQuery,
  useIncrementHydrationMutation,
  useLogMovementBreakMutation,
  useUpdateMoodMutation,
  useLogMeditationMutation,
} = apiSlice;