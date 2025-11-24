import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsBoolean, IsNumber, IsArray, IsEnum, Min, Max } from 'class-validator';
import { Type } from 'class-transformer';

export enum WellnessReminderType {
  HYDRATION = 'HYDRATION',
  MOVEMENT = 'MOVEMENT',
  POSTURE = 'POSTURE',
  EYE_REST = 'EYE_REST',
  MEDITATION = 'MEDITATION'
}

export class CreateWellnessReminderDto {
  @ApiProperty({
    description: 'Type of wellness reminder',
    enum: WellnessReminderType,
    example: WellnessReminderType.HYDRATION
  })
  @IsEnum(WellnessReminderType)
  type: WellnessReminderType;

  @ApiProperty({
    description: 'Whether the reminder is enabled',
    example: true,
    default: true
  })
  @IsBoolean()
  enabled: boolean;

  @ApiProperty({
    description: 'Frequency in minutes between reminders',
    example: 120,
    minimum: 5,
    maximum: 1440 // Max 24 hours
  })
  @IsNumber()
  @Min(5)
  @Max(1440)
  frequency: number;

  @ApiProperty({
    description: 'Start time in HH:mm format',
    example: '09:00'
  })
  @IsString()
  startTime: string;

  @ApiProperty({
    description: 'End time in HH:mm format',
    example: '18:00'
  })
  @IsString()
  endTime: string;

  @ApiProperty({
    description: 'Weekdays as array [1,2,3,4,5] for Mon-Fri',
    example: [1, 2, 3, 4, 5],
    isArray: true
  })
  @IsArray()
  @Type(() => Number)
  weekdays: number[];
}

export class UpdateWellnessReminderDto {
  @ApiProperty({
    description: 'Type of wellness reminder',
    enum: WellnessReminderType,
    example: WellnessReminderType.HYDRATION,
    required: false
  })
  @IsEnum(WellnessReminderType)
  @IsString()
  type?: WellnessReminderType;

  @ApiProperty({
    description: 'Whether the reminder is enabled',
    example: true,
    required: false
  })
  @IsBoolean()
  enabled?: boolean;

  @ApiProperty({
    description: 'Frequency in minutes between reminders',
    example: 120,
    minimum: 5,
    maximum: 1440,
    required: false
  })
  @IsNumber()
  @Min(5)
  @Max(1440)
  frequency?: number;

  @ApiProperty({
    description: 'Start time in HH:mm format',
    example: '09:00',
    required: false
  })
  @IsString()
  startTime?: string;

  @ApiProperty({
    description: 'End time in HH:mm format',
    example: '18:00',
    required: false
  })
  @IsString()
  endTime?: string;

  @ApiProperty({
    description: 'Weekdays as array [1,2,3,4,5] for Mon-Fri',
    example: [1, 2, 3, 4, 5],
    isArray: true,
    required: false
  })
  @IsArray()
  @Type(() => Number)
  weekdays?: number[];
}

export class WellnessReminderResponseDto {
  @ApiProperty({
    description: 'Reminder ID',
    example: 'clym8d1230000sbdp1234567'
  })
  id: string;

  @ApiProperty({
    description: 'User ID',
    example: 'clym8d1230000sbdp1234567'
  })
  userId: string;

  @ApiProperty({
    description: 'Type of wellness reminder',
    enum: WellnessReminderType,
    example: WellnessReminderType.HYDRATION
  })
  type: WellnessReminderType;

  @ApiProperty({
    description: 'Whether the reminder is enabled',
    example: true
  })
  enabled: boolean;

  @ApiProperty({
    description: 'Frequency in minutes between reminders',
    example: 120
  })
  frequency: number;

  @ApiProperty({
    description: 'Start time in HH:mm format',
    example: '09:00'
  })
  startTime: string;

  @ApiProperty({
    description: 'End time in HH:mm format',
    example: '18:00'
  })
  endTime: string;

  @ApiProperty({
    description: 'Weekdays as JSON array string',
    example: '[1,2,3,4,5]'
  })
  weekdays: string;

  @ApiProperty({
    description: 'Last time the reminder was triggered',
    example: '2024-01-15T14:00:00.000Z',
    required: false
  })
  lastTrigger?: Date;

  @ApiProperty({
    description: 'Creation timestamp',
    example: '2024-01-15T10:30:00.000Z'
  })
  createdAt: Date;

  @ApiProperty({
    description: 'Last update timestamp',
    example: '2024-01-15T15:45:00.000Z'
  })
  updatedAt: Date;
}