import { IsObject, ValidateNested } from 'class-validator';
import { Type } from 'class-transformer';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { UpdatePreferencesDto } from './update-preferences.dto';

export class UpdateProfileDto {
  @ApiProperty({
    description: 'User preferences object',
    type: UpdatePreferencesDto,
    example: {
      workDuration: 25,
      shortBreakDuration: 6,
      longBreakDuration: 15,
      longBreakInterval: 4,
      autoStartBreaks: false,
      autoStartWork: false,
      soundEnabled: true,
      volume: 70,
      ambientSound: 'forest',
      darkMode: false,
      notifications: {
        achievements: true,
        teamUpdates: true,
        weeklyReports: true,
        deadlineReminders: true,
        wellnessReminders: true,
      },
      wellness: {
        mindfulnessReminders: false,
        hydrationReminders: false,
        movementBreaks: false,
        eyeRest: true,
        endOfDay: false,
      },
    },
  })
  @IsObject()
  @ValidateNested()
  @Type(() => UpdatePreferencesDto)
  preferences: UpdatePreferencesDto;
}