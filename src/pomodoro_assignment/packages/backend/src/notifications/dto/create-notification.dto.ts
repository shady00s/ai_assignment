import { IsString, IsEnum, IsOptional, IsDateString } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export enum NotificationType {
  ACHIEVEMENT = 'ACHIEVEMENT',
  TEAM_UPDATE = 'TEAM_UPDATE',
  DEADLINE_REMINDER = 'DEADLINE_REMINDER',
  WELLNESS_REMINDER = 'WELLNESS_REMINDER',
  TASK_ASSIGNED = 'TASK_ASSIGNED',
  TASK_COMPLETED = 'TASK_COMPLETED',
  STREAK_MILESTONE = 'STREAK_MILESTONE',
  LEVEL_UP = 'LEVEL_UP',
}

export class CreateNotificationDto {
  @ApiProperty({
    description: 'Notification type',
    enum: NotificationType
  })
  @IsEnum(NotificationType)
  type: NotificationType;

  @ApiProperty({ description: 'Notification title' })
  @IsString()
  title: string;

  @ApiProperty({ description: 'Notification message/body' })
  @IsString()
  message: string;

  @ApiProperty({ description: 'Related entity ID (task, achievement, etc.)', required: false })
  @IsOptional()
  @IsString()
  entityId?: string;

  @ApiProperty({ description: 'Related entity type', required: false })
  @IsOptional()
  @IsString()
  entityType?: string;

  @ApiProperty({ description: 'Scheduled delivery time', required: false })
  @IsOptional()
  @IsDateString()
  scheduledFor?: string;

  @ApiProperty({ description: 'Additional notification data', required: false })
  @IsOptional()
  data?: Record<string, any>;
}