export { apiSlice } from './apiSlice';
export { tagTypes } from './apiSlice';

// Export all hooks
export {
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
  // useGetAchievementsQuery, // Disabled for now
  // useGetUserAchievementsQuery, // Disabled for now
  // useUnlockAchievementMutation, // Disabled for now
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
} from './apiSlice';