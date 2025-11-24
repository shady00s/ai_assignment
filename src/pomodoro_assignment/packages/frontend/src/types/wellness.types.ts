// Wellness Tracking System Types
// ====================================

export interface WellnessEntry {
  id: string;
  userId: string;
  date: string;

  // Hydration tracking
  hydrationGlasses: number;
  hydrationGoal: number;

  // Movement tracking
  movementBreaks: number;
  movementMinutes: number;
  stepsCount?: number;

  // Mental wellness
  meditationMinutes: number;
  breathingExercises: number;
  mindfulnessSessions: number;

  // Self-reported metrics
  moodRating: number;
  stressLevel: number;
  energyLevel: number;
  sleepQuality?: number;
  sleepHours?: number;

  // Session-based wellness
  postureChecks: number;
  eyeRestBreaks: number;

  createdAt: string;
  updatedAt: string;

  // Computed fields from backend
  hydrationProgress?: number;
  wellnessScore?: number;
}

export interface WellnessReminder {
  id: string;
  userId: string;
  type: 'HYDRATION' | 'MOVEMENT' | 'POSTURE' | 'EYE_REST' | 'MEDITATION';
  enabled: boolean;
  frequency: number;
  startTime: string;
  endTime: string;
  weekdays: number[];
  lastTrigger?: string;
  createdAt: string;
  updatedAt: string;
}

export interface WellnessGoal {
  id: string;
  userId: string;
  category: 'HYDRATION' | 'MOVEMENT' | 'MEDITATION' | 'SLEEP';
  targetValue: number;
  period: 'DAILY' | 'WEEKLY' | 'MONTHLY';
  active: boolean;
  createdAt: string;
  updatedAt: string;

  // Computed fields from backend
  currentProgress?: number;
  progressPercentage?: number;
}

export interface WellnessAnalytics {
  mindfulnessMinutes: number;
  hydrationGoal: number;
  hydrationCurrent: number;
  movementGoal: number;
  movementCurrent: number;
  moodRating: number;
  stressLevel: number;
  energyLevel: number;
}

// Extended analytics structure for detailed wellness analytics
export interface DetailedWellnessAnalytics {
  userId: string;
  period: number;
  startDate: string;
  endDate: string;
  hydration: {
    weeklyAverage: number;
    bestDay: string;
    consistencyScore: number;
    trend: 'improving' | 'stable' | 'declining';
    goalAchievementRate: number;
  };
  movement: {
    averageBreaks: number;
    averageMinutes: number;
    mostActiveDay: string;
    weeklyTotal: number;
    goalAchievementRate: number;
  };
  mentalWellness: {
    averageMoodRating: number;
    averageStressLevel: number;
    averageEnergyLevel: number;
    meditationStreak: number;
    totalMindfulnessSessions: number;
  };
  sleep: {
    averageHours: number;
    averageQuality: number;
    consistencyScore: number;
    bestSleepDay: string;
  };
  overall: {
    overallScore: number;
    trendDirection: 'upward' | 'stable' | 'downward';
    streakDays: number;
    perfectDaysCount: number;
    complianceRate: number;
  };
  recommendations?: WellnessRecommendation[];
  trends?: WellnessTrend[];
}

export interface WellnessRecommendation {
  id: string;
  type: 'HYDRATION' | 'MOVEMENT' | 'MENTAL_WELLNESS';
  title: string;
  description: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH';
  actionable: boolean;
  estimatedImpact: string;
}

export interface WellnessTrend {
  date: string;
  hydrationGlasses: number;
  movementBreaks: number;
  moodRating: number;
  stressLevel: number;
  energyLevel: number;
  wellnessScore: number;
}

export interface Recommendation {
  id: string;
  type: 'HYDRATION' | 'MOVEMENT' | 'MEDITATION' | 'SLEEP' | 'POSTURE' | 'STRESS';
  title: string;
  description: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH';
  actionable: boolean;
  icon: string;
  category: string;
}

export interface MeditationOption {
  id: string;
  name: string;
  duration: number;
  type: 'GUIDED' | 'BREATHING' | 'MINDFULNESS';
  audioUrl?: string;
  description?: string;
}

export interface MovementType {
  id: string;
  name: string;
  icon: string;
  category: 'CARDIO' | 'STRENGTH' | 'FLEXIBILITY' | 'BALANCE' | 'SPORTS';
  metValue?: number; // Metabolic equivalent for calorie calculation
}

export interface HydrationProgress {
  currentGlasses: number;
  dailyGoal: number;
  percentageComplete: number;
  totalMl: number;
  streakDays: number;
  lastGlassesTime?: string;
}

export interface MovementProgress {
  currentBreaks: number;
  dailyGoal: number;
  totalMinutes: number;
  stepsCount?: number;
  caloriesBurned?: number;
  activeMinutes: number;
  streakDays: number;
}

export interface MoodProgress {
  currentMood: number;
  currentStress: number;
  currentEnergy: number;
  lastCheckIn: string;
  moodTrend: 'IMPROVING' | 'DECLINING' | 'STABLE';
  checkInStreak: number;
}

export interface MeditationProgress {
  totalMinutes: number;
  sessionGoal: number;
  completedSessions: number;
  averageSessionLength: number;
  streakDays: number;
  favoriteType: string;
}

// Component Props Interfaces
export interface HydrationTrackerProps {
  currentGlasses: number;
  dailyGoal: number;
  glassSize: number;
  onIncrement: () => void;
  onDecrement: () => void;
  onGoalUpdate: (newGoal: number) => void;
  isLoading: boolean;
  compact?: boolean;
}

export interface MovementTrackerProps {
  movementBreaks: number;
  movementMinutes: number;
  stepsCount?: number;
  dailyGoal: number;
  onStartBreak: () => void;
  onEndBreak: (duration: number) => void;
  onLogActivity: (minutes: number, type: string) => void;
  isLoading: boolean;
  compact?: boolean;
}

export interface MoodTrackerProps {
  moodRating: number;
  stressLevel: number;
  energyLevel: number;
  onMoodUpdate: (mood: number) => void;
  onStressUpdate: (stress: number) => void;
  onEnergyUpdate: (energy: number) => void;
  lastCheckIn: string;
  isLoading: boolean;
  compact?: boolean;
}

export interface MeditationTimerProps {
  totalMinutes: number;
  sessionGoal: number;
  onStartSession: (duration: number) => void;
  onCompleteSession: (duration: number, quality: number) => void;
  guidedOptions: MeditationOption[];
  isLoading: boolean;
  compact?: boolean;
}

export interface WellnessDashboardProps {
  date?: string;
  viewMode: 'compact' | 'detailed' | 'analytics';
  onDateChange?: (date: string) => void;
  className?: string;
}

export interface WellnessAnalyticsProps {
  analytics: WellnessAnalytics;
  history: WellnessEntry[];
  days: number;
  isLoading: boolean;
}

// Notification System Types
export interface WellnessNotification {
  id: string;
  type: WellnessReminder['type'];
  title: string;
  message: string;
  scheduledTime: string;
  isRead: boolean;
  actionUrl?: string;
  actionText?: string;
}

export interface NotificationSettings {
  hydration: boolean;
  movement: boolean;
  posture: boolean;
  eyeRest: boolean;
  meditation: boolean;
  sound: boolean;
  vibration: boolean;
  desktop: boolean;
}

// API Request/Response Types
export interface CreateWellnessEntryRequest {
  date?: string;
  hydrationGlasses?: number;
  hydrationGoal?: number;
  movementBreaks?: number;
  movementMinutes?: number;
  stepsCount?: number;
  meditationMinutes?: number;
  breathingExercises?: number;
  mindfulnessSessions?: number;
  moodRating?: number;
  stressLevel?: number;
  energyLevel?: number;
  sleepQuality?: number;
  sleepHours?: number;
  postureChecks?: number;
  eyeRestBreaks?: number;
}

export interface UpdateWellnessEntryRequest extends Partial<CreateWellnessEntryRequest> {
  date: string;
}

export interface IncrementHydrationRequest {
  glasses: number;
}

export interface LogMovementRequest {
  duration: number;
  type: string;
  intensity?: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface UpdateMoodRequest {
  mood: number;
  stress: number;
  energy: number;
}

export interface LogMeditationRequest {
  minutes: number;
  type: string;
  quality: number;
  notes?: string;
}

export interface CreateWellnessReminderRequest {
  type: WellnessReminder['type'];
  enabled: boolean;
  frequency: number;
  startTime: string;
  endTime: string;
  weekdays: number[];
}

export interface UpdateWellnessGoalRequest {
  targetValue: number;
  period: WellnessGoal['period'];
  active: boolean;
}

// Error Handling Types
export interface WellnessError {
  code: string;
  message: string;
  field?: string;
  details?: any;
}

// Utility Types
export type WellnessCategory =
  | 'HYDRATION'
  | 'MOVEMENT'
  | 'MEDITATION'
  | 'SLEEP'
  | 'POSTURE'
  | 'EYE_REST'
  | 'STRESS'
  | 'ENERGY';

export type TrendDirection = 'IMPROVING' | 'DECLINING' | 'STABLE';

export type PriorityLevel = 'LOW' | 'MEDIUM' | 'HIGH';

export type ViewMode = 'compact' | 'detailed' | 'analytics';

export type MeditationType = 'GUIDED' | 'BREATHING' | 'MINDFULNESS';

export type MovementCategory = 'CARDIO' | 'STRENGTH' | 'FLEXIBILITY' | 'BALANCE' | 'SPORTS';

export type MoodRating = 1 | 2 | 3 | 4 | 5;

export type StressRating = 1 | 2 | 3 | 4 | 5;

export type EnergyRating = 1 | 2 | 3 | 4 | 5;

// Chart Data Types
export interface WellnessChartData {
  date: string;
  hydration: number;
  movement: number;
  mood: number;
  stress: number;
  energy: number;
  meditation: number;
}

export interface TrendData {
  period: string;
  value: number;
  goal?: number;
}

// Local Storage Types
export interface WellnessCache {
  todayEntry: WellnessEntry | null;
  lastSync: string;
  offlineActions: WellnessAction[];
}

export interface WellnessAction {
  type: 'INCREMENT_HYDRATION' | 'LOG_MOVEMENT' | 'UPDATE_MOOD' | 'LOG_MEDITATION';
  payload: any;
  timestamp: string;
  synced: boolean;
}