import { IsOptional, IsBoolean, IsNumber, IsEnum, Min, Max, IsObject, ValidateNested } from 'class-validator';
import { Type } from 'class-transformer';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

// Nested DTO for notification preferences
export class NotificationsPreferencesDto {
  @ApiPropertyOptional({
    description: 'Achievement notifications enabled',
    example: true,
  })
  @IsOptional()
  @IsBoolean()
  achievements?: boolean;

  @ApiPropertyOptional({
    description: 'Team update notifications enabled',
    example: true,
  })
  @IsOptional()
  @IsBoolean()
  teamUpdates?: boolean;

  @ApiPropertyOptional({
    description: 'Weekly report notifications enabled',
    example: true,
  })
  @IsOptional()
  @IsBoolean()
  weeklyReports?: boolean;

  @ApiPropertyOptional({
    description: 'Deadline reminder notifications enabled',
    example: true,
  })
  @IsOptional()
  @IsBoolean()
  deadlineReminders?: boolean;

  @ApiPropertyOptional({
    description: 'Wellness reminder notifications enabled',
    example: true,
  })
  @IsOptional()
  @IsBoolean()
  wellnessReminders?: boolean;
}

// Nested DTO for wellness preferences
export class WellnessPreferencesDto {
  @ApiPropertyOptional({
    description: 'Mindfulness reminders enabled',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  mindfulnessReminders?: boolean;

  @ApiPropertyOptional({
    description: 'Hydration reminders enabled',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  hydrationReminders?: boolean;

  @ApiPropertyOptional({
    description: 'Movement break reminders enabled',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  movementBreaks?: boolean;

  @ApiPropertyOptional({
    description: 'Eye rest reminders enabled',
    example: true,
  })
  @IsOptional()
  @IsBoolean()
  eyeRest?: boolean;

  @ApiPropertyOptional({
    description: 'End of day summary enabled',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  endOfDay?: boolean;
}

export class UpdatePreferencesDto {
  @ApiPropertyOptional({
    description: 'Work duration in minutes',
    example: 25,
    minimum: 1,
    maximum: 60,
  })
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(60)
  workDuration?: number;

  @ApiPropertyOptional({
    description: 'Short break duration in minutes',
    example: 5,
    minimum: 1,
    maximum: 15,
  })
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(15)
  shortBreakDuration?: number;

  @ApiPropertyOptional({
    description: 'Long break duration in minutes',
    example: 15,
    minimum: 1,
    maximum: 30,
  })
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(30)
  longBreakDuration?: number;

  @ApiPropertyOptional({
    description: 'Long break interval (number of work sessions)',
    example: 4,
    minimum: 2,
    maximum: 10,
  })
  @IsOptional()
  @IsNumber()
  @Min(2)
  @Max(10)
  longBreakInterval?: number;

  @ApiPropertyOptional({
    description: 'Auto-start breaks',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  autoStartBreaks?: boolean;

  @ApiPropertyOptional({
    description: 'Auto-start work sessions',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  autoStartWork?: boolean;

  @ApiPropertyOptional({
    description: 'Sound enabled',
    example: true,
  })
  @IsOptional()
  @IsBoolean()
  soundEnabled?: boolean;

  @ApiPropertyOptional({
    description: 'Volume level (0-100)',
    example: 70,
    minimum: 0,
    maximum: 100,
  })
  @IsOptional()
  @IsNumber()
  @Min(0)
  @Max(100)
  volume?: number;

  @ApiPropertyOptional({
    description: 'Ambient sound type',
    enum: ['forest', 'ocean', 'cafe', 'rain', 'none'],
    example: 'forest',
  })
  @IsOptional()
  @IsEnum(['forest', 'ocean', 'cafe', 'rain', 'none'])
  ambientSound?: string;

  @ApiPropertyOptional({
    description: 'Dark mode enabled',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  darkMode?: boolean;

  @ApiPropertyOptional({
    description: 'Notification preferences',
    type: NotificationsPreferencesDto,
  })
  @IsOptional()
  @IsObject()
  @ValidateNested()
  @Type(() => NotificationsPreferencesDto)
  notifications?: NotificationsPreferencesDto;

  @ApiPropertyOptional({
    description: 'Wellness preferences',
    type: WellnessPreferencesDto,
  })
  @IsOptional()
  @IsObject()
  @ValidateNested()
  @Type(() => WellnessPreferencesDto)
  wellness?: WellnessPreferencesDto;
}