import { ApiProperty } from '@nestjs/swagger';
import { AchievementDto } from './achievement.dto';

export class UserAchievementDto {
  @ApiProperty({ description: 'User achievement ID' })
  id: string;

  @ApiProperty({ description: 'User ID' })
  userId: string;

  @ApiProperty({ description: 'Achievement ID' })
  achievementId: string;

  @ApiProperty({
    description: 'Achievement details',
    type: AchievementDto,
  })
  achievement: AchievementDto;

  @ApiProperty({
    description: 'When achievement was unlocked',
    example: '2024-01-15T10:00:00.000Z',
  })
  unlockedAt: string;

  @ApiProperty({
    description: 'Progress percentage (0-100)',
    example: 75,
    minimum: 0,
    maximum: 100,
  })
  progress: number;
}

export class UnlockAchievementDto {
  @ApiProperty({
    description: 'Progress towards achievement (0-100)',
    example: 100,
    minimum: 0,
    maximum: 100,
    required: false,
  })
  progress?: number;
}