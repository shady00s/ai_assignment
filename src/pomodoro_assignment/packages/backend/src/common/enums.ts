// Enums for SQLite compatibility (stored as strings in database, but provide type safety in TypeScript)

export enum TeamRole {
  OWNER = 'OWNER',
  ADMIN = 'ADMIN',
  MEMBER = 'MEMBER',
}

export enum TaskStatus {
  TODO = 'TODO',
  IN_PROGRESS = 'IN_PROGRESS',
  COMPLETED = 'COMPLETED',
  CANCELLED = 'CANCELLED',
}

export enum TaskPriority {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  URGENT = 'URGENT',
}

export enum SessionType {
  POMODORO = 'POMODORO',
  SHORT_BREAK = 'SHORT_BREAK',
  LONG_BREAK = 'LONG_BREAK',
  CUSTOM = 'CUSTOM',
}

export enum AchievementCategory {
  FOCUS = 'FOCUS',
  CONSISTENCY = 'CONSISTENCY',
  WELLNESS = 'WELLNESS',
  COLLABORATION = 'COLLABORATION',
  MILESTONES = 'MILESTONES',
}

export enum ChallengeType {
  FOCUS_TIME = 'FOCUS_TIME',
  TASK_COMPLETION = 'TASK_COMPLETION',
  WELLNESS_SCORE = 'WELLNESS_SCORE',
  TEAM_COLLABORATION = 'TEAM_COLLABORATION',
}

export enum NotificationType {
  INFO = 'INFO',
  SUCCESS = 'SUCCESS',
  WARNING = 'WARNING',
  ERROR = 'ERROR',
  ACHIEVEMENT = 'ACHIEVEMENT',
  TEAM_UPDATE = 'TEAM_UPDATE',
  REMINDER = 'REMINDER',
}

export enum NotificationPriority {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  URGENT = 'URGENT',
}

// Helper functions for enum validation
export const isValidTeamRole = (value: string): value is TeamRole =>
  Object.values(TeamRole).includes(value as TeamRole);

export const isValidTaskStatus = (value: string): value is TaskStatus =>
  Object.values(TaskStatus).includes(value as TaskStatus);

export const isValidTaskPriority = (value: string): value is TaskPriority =>
  Object.values(TaskPriority).includes(value as TaskPriority);

export const isValidSessionType = (value: string): value is SessionType =>
  Object.values(SessionType).includes(value as SessionType);

export const isValidAchievementCategory = (value: string): value is AchievementCategory =>
  Object.values(AchievementCategory).includes(value as AchievementCategory);

export const isValidChallengeType = (value: string): value is ChallengeType =>
  Object.values(ChallengeType).includes(value as ChallengeType);

export const isValidNotificationType = (value: string): value is NotificationType =>
  Object.values(NotificationType).includes(value as NotificationType);

export const isValidNotificationPriority = (value: string): value is NotificationPriority =>
  Object.values(NotificationPriority).includes(value as NotificationPriority);