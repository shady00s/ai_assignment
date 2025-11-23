// Import and re-export frontend types for backend compatibility
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

// Extended Task type with parsed JSON fields and relations
export type TaskWithParsedFields = Omit<Task, 'tags'> & {
  tags: string[];
  creator?: {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
    avatar?: string;
  };
  assignee?: {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
    avatar?: string;
  };
  team?: {
    id: string;
    name: string;
  };
  dependencies?: {
    id: string;
    prerequisite: {
      id: string;
      title: string;
    };
  }[];
  dependents?: {
    id: string;
    dependentTask: {
      id: string;
      title: string;
    };
  }[];
  sessions?: {
    id: string;
    duration: number;
    startTime: string;
    endTime: string;
    quality: number;
  }[];
}

// Prisma Task type with relations
export type PrismaTask = {
  id: string;
  title: string;
  description: string | null;
  priority: string;
  status: string;
  dueDate: Date | null;
  estimatedPomodoros: number;
  completedPomodoros: number;
  estimatedMinutes: number | null;
  actualMinutes: number | null;
  assigneeId: string | null;
  creatorId: string;
  teamId: string | null;
  tags: string | null;
  complexity: number;
  completedAt: Date | null;
  createdAt: Date;
  updatedAt: Date;
  // Relations would be added here as needed
} & {
  creator?: {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
    avatar?: string;
  };
  assignee?: {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
    avatar?: string;
  };
  team?: {
    id: string;
    name: string;
  };
};