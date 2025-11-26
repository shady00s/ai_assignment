// Wellness Entry DTOs
export { CreateWellnessEntryDto } from './create-wellness-entry.dto';
export { UpdateWellnessEntryDto } from './update-wellness-entry.dto';
export { WellnessEntryResponseDto } from './wellness-entry-response.dto';

// Wellness Reminder DTOs
export {
  CreateWellnessReminderDto,
  UpdateWellnessReminderDto,
  WellnessReminderResponseDto,
  WellnessReminderType
} from './wellness-reminder.dto';

// Wellness Goal DTOs
export {
  CreateWellnessGoalDto,
  UpdateWellnessGoalDto,
  WellnessGoalResponseDto,
  WellnessGoalCategory,
  WellnessGoalPeriod
} from './wellness-goal.dto';

// Wellness Analytics DTOs
export {
  WellnessAnalyticsDto,
  WellnessTrendsDto,
  WellnessRecommendationDto
} from './wellness-analytics.dto';

// Query DTOs
export { WellnessHistoryQueryDto, WellnessAnalyticsQueryDto } from './wellness-query.dto';

// Hydration DTOs
export {
  IncrementHydrationDto,
  SetHydrationGoalDto,
  QuickWaterDto
} from './hydration.dto';

// Movement DTOs
export {
  LogMovementDto,
  LogStepsDto,
  SetMovementGoalDto,
  QuickMovementDto,
  MovementIntensity
} from './movement.dto';

// Mood DTOs
export {
  UpdateMoodDto,
  LogSleepDto,
  QuickMoodDto
} from './mood.dto';

// Meditation DTOs
export {
  LogMeditationDto,
  CompleteMeditationDto,
  LogBreathingDto,
  QuickMeditationDto
} from './meditation.dto';

// Posture and Eye Rest DTOs
export {
  LogPostureCheckDto,
  LogEyeRestDto
} from './posture.dto';

 

// Utility DTOs
export {
  AcknowledgeRecommendationDto,
  WellnessScoreResponseDto,
  WellnessAchievementDto
} from './utility.dto';