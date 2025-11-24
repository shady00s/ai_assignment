import { ApiProperty } from '@nestjs/swagger';

export class WellnessEntryResponseDto {
  @ApiProperty({
    description: 'Wellness entry ID',
    example: 'clym8d1230000sbdp1234567'
  })
  id: string;

  @ApiProperty({
    description: 'User ID',
    example: 'clym8d1230000sbdp1234567'
  })
  userId: string;

  @ApiProperty({
    description: 'Entry date',
    example: '2024-01-15T00:00:00.000Z'
  })
  date: Date;

  // Hydration tracking
  @ApiProperty({
    description: 'Number of glasses of water consumed',
    example: 6
  })
  hydrationGlasses: number;

  @ApiProperty({
    description: 'Daily hydration goal in glasses',
    example: 8
  })
  hydrationGoal: number;

  // Movement tracking
  @ApiProperty({
    description: 'Number of movement breaks taken',
    example: 4
  })
  movementBreaks: number;

  @ApiProperty({
    description: 'Total movement minutes',
    example: 25
  })
  movementMinutes: number;

  @ApiProperty({
    description: 'Step count',
    example: 8500,
    required: false
  })
  stepsCount?: number;

  // Mental wellness
  @ApiProperty({
    description: 'Time spent meditating in minutes',
    example: 15
  })
  meditationMinutes: number;

  @ApiProperty({
    description: 'Number of breathing exercises completed',
    example: 3
  })
  breathingExercises: number;

  @ApiProperty({
    description: 'Number of mindfulness sessions',
    example: 2
  })
  mindfulnessSessions: number;

  // Self-reported metrics
  @ApiProperty({
    description: 'Mood rating (1=very poor, 5=excellent)',
    example: 4
  })
  moodRating: number;

  @ApiProperty({
    description: 'Stress level (1=very low, 5=very high)',
    example: 2
  })
  stressLevel: number;

  @ApiProperty({
    description: 'Energy level (1=very low, 5=very high)',
    example: 4
  })
  energyLevel: number;

  @ApiProperty({
    description: 'Sleep quality rating (1=very poor, 5=excellent)',
    example: 4,
    required: false
  })
  sleepQuality?: number;

  @ApiProperty({
    description: 'Hours of sleep',
    example: 7.5,
    required: false
  })
  sleepHours?: number;

  // Session-based wellness
  @ApiProperty({
    description: 'Number of posture checks completed',
    example: 6
  })
  postureChecks: number;

  @ApiProperty({
    description: 'Number of eye rest breaks taken',
    example: 4
  })
  eyeRestBreaks: number;

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

  // Computed fields
  @ApiProperty({
    description: 'Hydration progress percentage',
    example: 75
  })
  hydrationProgress: number;

  @ApiProperty({
    description: 'Overall wellness score (0-100)',
    example: 82
  })
  wellnessScore: number;
}