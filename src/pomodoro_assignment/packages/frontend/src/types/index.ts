// User types
export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  avatar?: string;
  teamId?: string;
  level: number;
  xp: number;
  streak: number;
  totalFocusTime: number; // in minutes
  tasksCompleted: number;
  qualityScore: number;
  wellnessScore: number;
  preferences: UserPreferences;
  createdAt: string;
  updatedAt: string;
}

export interface UserPreferences {
  workDuration: number; // in minutes
  shortBreakDuration: number;
  longBreakDuration: number;
  longBreakInterval: number; // after how many work sessions
  autoStartBreaks: boolean;
  autoStartWork: boolean;
  soundEnabled: boolean;
  volume: number; // 0-100
  ambientSound: 'forest' | 'ocean' | 'cafe' | 'rain' | 'none';
  darkMode: boolean;
  notifications: NotificationPreferences;
  wellness: WellnessPreferences;
}

export interface NotificationPreferences {
  achievements: boolean;
  teamUpdates: boolean;
  weeklyReports: boolean;
  deadlineReminders: boolean;
  wellnessReminders: boolean;
}

export interface WellnessPreferences {
  mindfulnessReminders: boolean;
  hydrationReminders: boolean;
  movementBreaks: boolean;
  eyeRest: boolean;
  endOfDay: boolean;
}

// Task types
export interface Task {
  id: string;
  title: string;
  description?: string;
  status: TaskStatus;
  priority: TaskPriority;
  estimatedPomodoros: number;
  completedPomodoros: number;
  assigneeId?: string;
  assignee?: User;
  projectId?: string;
  dueDate?: string;
  createdAt: string;
  updatedAt: string;
  completedAt?: string;
  tags: string[];
}

export type TaskStatus = 'TODO' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED';
export type TaskPriority = 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';

// Session types
export interface Session {
  id: string;
  type: SessionType;
  startTime: string;
  endTime?: string;
  duration: number; // in minutes
  taskId?: string;
  task?: Task;
  userId: string;
  user: User;
  quality?: number; // 1-5 rating
  notes?: string;
  interruptions: number;
  completed: boolean;
  createdAt: string;
}

export type SessionType = 'POMODORO' | 'SHORT_BREAK' | 'LONG_BREAK';

// Team types
export interface Team {
  id: string;
  name: string;
  description?: string;
  avatar?: string;
  ownerId: string;
  members: TeamMember[];
  challenges: Challenge[];
  createdAt: string;
  updatedAt: string;
}

export interface TeamMember {
  id: string;
  userId: string;
  user: User;
  role: TeamRole;
  joinedAt: string;
}

export type TeamRole = 'OWNER' | 'ADMIN' | 'MEMBER';

// Achievement types
export interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: AchievementCategory;
  requirement: AchievementRequirement;
  xpReward: number;
  badgeUrl?: string;
  createdAt: string;
}

export type AchievementCategory = 'FOCUS' | 'CONSISTENCY' | 'WELLNESS' | 'COLLABORATION' | 'MILESTONES';

export interface AchievementRequirement {
  type: 'SESSION_COUNT' | 'STREAK_DAYS' | 'TOTAL_TIME' | 'TASKS_COMPLETED' | 'TEAM_HELP';
  value: number;
  timeframe?: 'DAILY' | 'WEEKLY' | 'MONTHLY' | 'ALL_TIME';
}

export interface UserAchievement {
  id: string;
  userId: string;
  achievementId: string;
  achievement: Achievement;
  unlockedAt: string;
  progress: number; // 0-100 percentage
}

// Challenge types
export interface Challenge {
  id: string;
  name: string;
  description: string;
  type: ChallengeType;
  targetValue: number;
  currentValue: number;
  startDate: string;
  endDate: string;
  participantIds: string[];
  participants: User[];
  rewards: ChallengeReward;
  createdBy: string;
  createdAt: string;
}

export type ChallengeType = 'FOCUS_TIME' | 'TASK_COMPLETION' | 'WELLNESS_SCORE' | 'TEAM_COLLABORATION';

export interface ChallengeReward {
  xp: number;
  badge?: string;
  customReward?: string;
}

// Analytics types
export interface FocusAnalytics {
  dailyFocusTime: number; // minutes today
  weeklyFocusTime: number; // minutes this week
  monthlyFocusTime: number; // minutes this month
  averageSessionLength: number;
  peakFocusHours: number[]; // hours of day
  focusTrend: 'IMPROVING' | 'DECLINING' | 'STABLE';
  completionRate: number; // percentage
}

export interface WellnessAnalytics {
  mindfulnessMinutes: number;
  hydrationGoal: number;
  hydrationCurrent: number;
  movementGoal: number;
  movementCurrent: number;
  moodRating: number; // 1-5
  stressLevel: number; // 1-5
  energyLevel: number; // 1-5
}

export interface TeamAnalytics {
  teamId: string;
  teamName: string;
  memberCount: number;
  totalFocusTime: number; // total minutes for team in date range
  averageFocusTime: number; // average minutes per member
  tasksCompleted: number;
  averageCompletionRate: number; // percentage
  topPerformers: TeamMemberStats[];
  focusTrend: 'IMPROVING' | 'DECLINING' | 'STABLE';
  wellnessScore: number; // team average
  collaborationScore: number; // based on cross-team tasks
  period: {
    startDate: string;
    endDate: string;
  };
}

export interface TeamMemberStats {
  userId: string;
  user: User;
  focusTime: number; // minutes in period
  tasksCompleted: number;
  completionRate: number;
  wellnessScore: number;
  streakDays: number;
}

// API Request/Response types
export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  user: User;
  token: string;
  refreshToken: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  teamId?: string;
}

export interface RegisterResponse {
  user: User;
  token: string;
  refreshToken: string;
}

export interface CreateTaskRequest {
  title: string;
  description?: string;
  priority: TaskPriority;
  estimatedPomodoros: number;
  assigneeId?: string;
  dueDate?: string;
  tags: string[];
}

export interface UpdateTaskRequest {
  title?: string;
  description?: string;
  status?: TaskStatus;
  priority?: TaskPriority;
  estimatedPomodoros?: number;
  assigneeId?: string;
  dueDate?: string;
  tags?: string[];
}

export interface CreateSessionRequest {
  type: SessionType;
  taskId?: string;
  plannedDuration?: number;
}

export interface CompleteSessionRequest {
  quality?: number;
  notes?: string;
  interruptions?: number;
}

// UI State types
export interface TimerState {
  isRunning: boolean;
  isPaused: boolean;
  remainingTime: number; // in seconds
  totalTime: number; // in seconds
  sessionType: SessionType;
  currentSession?: Session;
  sessionsCompleted: number;
}

export interface UIState {
  sidebarOpen: boolean;
  theme: 'light' | 'dark' | 'auto';
  currentView: string;
  loading: boolean;
  error?: string;
  notifications: Notification[];
}

export interface Notification {
  id: string;
  type: 'INFO' | 'SUCCESS' | 'WARNING' | 'ERROR' | 'ACHIEVEMENT' | 'TEAM_UPDATE' | 'REMINDER';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';
  userId: string;
  senderId?: string; // for team notifications
  action?: {
    label: string;
    url?: string;
    data?: any;
  };
  metadata?: {
    relatedEntityType?: 'task' | 'session' | 'achievement' | 'team' | 'challenge';
    relatedEntityId?: string;
  };
}

// Filter and Sort types
export interface TaskFilters {
  status?: TaskStatus[];
  priority?: TaskPriority[];
  assigneeId?: string[];
  tags?: string[];
  dueDateRange?: {
    start: string;
    end: string;
  };
}

export interface TaskSort {
  field: 'createdAt' | 'updatedAt' | 'dueDate' | 'priority' | 'title' | 'status';
  direction: 'ASC' | 'DESC';
}

export interface SessionFilters {
  type?: SessionType[];
  dateRange?: {
    start: string;
    end: string;
  };
  userId?: string[];
  taskId?: string[];
}