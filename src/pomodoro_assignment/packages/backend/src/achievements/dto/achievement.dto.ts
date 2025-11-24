import { ApiProperty } from '@nestjs/swagger';

export class AchievementRequirementDto {
  @ApiProperty({
    description: 'Type of achievement requirement',
    enum: ['SESSION_COUNT', 'STREAK_DAYS', 'TOTAL_TIME', 'TASKS_COMPLETED', 'TEAM_HELP'],
    example: 'SESSION_COUNT',
  })
  type: 'SESSION_COUNT' | 'STREAK_DAYS' | 'TOTAL_TIME' | 'TASKS_COMPLETED' | 'TEAM_HELP';

  @ApiProperty({
    description: 'Target value to achieve',
    example: 100,
    minimum: 1,
  })
  value: number;

  @ApiProperty({
    description: 'Timeframe for achievement',
    enum: ['DAILY', 'WEEKLY', 'MONTHLY', 'ALL_TIME'],
    example: 'ALL_TIME',
    required: false,
  })
  timeframe?: 'DAILY' | 'WEEKLY' | 'MONTHLY' | 'ALL_TIME';
}

export class AchievementDto {
  @ApiProperty({ description: 'Achievement ID' })
  id: string;

  @ApiProperty({
    description: 'Achievement name',
    example: 'Focus Master',
  })
  name: string;

  @ApiProperty({
    description: 'Achievement description',
    example: 'Complete 100 focus sessions',
  })
  description: string;

  @ApiProperty({
    description: 'Achievement icon or emoji',
    example: '🎯',
  })
  icon: string;

  @ApiProperty({
    description: 'Achievement category',
    enum: ['FOCUS', 'CONSISTENCY', 'WELLNESS', 'COLLABORATION', 'MILESTONES'],
    example: 'FOCUS',
  })
  category: 'FOCUS' | 'CONSISTENCY' | 'WELLNESS' | 'COLLABORATION' | 'MILESTONES';

  @ApiProperty({
    description: 'Achievement requirements',
    type: AchievementRequirementDto,
  })
  requirement: AchievementRequirementDto;

  @ApiProperty({
    description: 'XP reward for unlocking achievement',
    example: 50,
    minimum: 0,
  })
  xpReward: number;

  @ApiProperty({
    description: 'Achievement badge URL',
    example: 'https://example.com/badges/focus-master.png',
    required: false,
  })
  badgeUrl?: string;

  @ApiProperty({
    description: 'Whether achievement is active',
    example: true,
  })
  isActive: boolean;

  @ApiProperty({
    description: 'Achievement creation date',
    example: '2024-01-15T10:00:00.000Z',
  })
  createdAt: string;
}