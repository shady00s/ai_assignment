import { ApiProperty } from '@nestjs/swagger';
import { IsBoolean, IsString, IsOptional } from 'class-validator';

export class AcknowledgeRecommendationDto {
  @ApiProperty({
    description: 'Whether the recommendation has been acknowledged',
    example: true
  })
  @IsBoolean()
  acknowledged: boolean;
}

export class WellnessScoreResponseDto {
  @ApiProperty({
    description: 'Overall wellness score (0-100)',
    example: 82
  })
  score: number;

  @ApiProperty({
    description: 'Detailed breakdown of wellness score components',
    example: {
      hydration: 75,
      movement: 90,
      mentalWellness: 85,
      mindfulness: 70,
      postureEyeRest: 88
    }
  })
  breakdown: {
    hydration: number;
    movement: number;
    mentalWellness: number;
    mindfulness: number;
    postureEyeRest: number;
  };

  @ApiProperty({
    description: 'Current trend direction',
    example: 'upward',
    enum: ['upward', 'stable', 'downward']
  })
  trend: string;
}

export class WellnessAchievementDto {
  @ApiProperty({
    description: 'Achievement ID',
    example: 'wellness_001'
  })
  id: string;

  @ApiProperty({
    description: 'Achievement title',
    example: 'Hydration Hero'
  })
  title: string;

  @ApiProperty({
    description: 'Achievement description',
    example: 'Reached your hydration goal for 7 consecutive days'
  })
  description: string;

  @ApiProperty({
    description: 'Category of wellness achievement',
    example: 'HYDRATION',
    enum: ['HYDRATION', 'MOVEMENT', 'MEDITATION', 'SLEEP', 'CONSISTENCY']
  })
  category: string;

  @ApiProperty({
    description: 'When the achievement was unlocked',
    example: '2024-01-15T10:30:00.000Z'
  })
  unlockedAt: string;

  @ApiProperty({
    description: 'Current progress towards the achievement (0-100)',
    example: 100
  })
  progress: number;

  @ApiProperty({
    description: 'URL to achievement badge/icon',
    example: '/assets/achievements/hydration-hero.svg'
  })
  badgeUrl?: string;

  @ApiProperty({
    description: 'XP reward for the achievement',
    example: 50
  })
  xpReward: number;
}