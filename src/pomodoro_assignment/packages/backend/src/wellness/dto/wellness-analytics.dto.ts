import { ApiProperty } from '@nestjs/swagger';

export class WellnessAnalyticsDto {
  @ApiProperty({
    description: 'User ID',
    example: 'clym8d1230000sbdp1234567'
  })
  userId: string;

  @ApiProperty({
    description: 'Analysis period in days',
    example: 30,
    minimum: 1,
    maximum: 365
  })
  period: number;

  @ApiProperty({
    description: 'Start date of analysis period',
    example: '2024-01-01T00:00:00.000Z'
  })
  startDate: Date;

  @ApiProperty({
    description: 'End date of analysis period',
    example: '2024-01-31T23:59:59.999Z'
  })
  endDate: Date;

  // Hydration analytics
  @ApiProperty({
    description: 'Hydration analytics',
    type: 'object',
    properties: {
      weeklyAverage: { type: 'number', example: 6.8 },
      bestDay: { type: 'string', example: 'Wednesday' },
      consistencyScore: { type: 'number', example: 0.75 },
      trend: { type: 'string', example: 'improving' },
      goalAchievementRate: { type: 'number', example: 0.65 }
    }
  })
  hydration: {
    weeklyAverage: number;
    bestDay: string;
    consistencyScore: number;
    trend: 'improving' | 'stable' | 'declining';
    goalAchievementRate: number;
  };

  // Movement analytics
  @ApiProperty({
    description: 'Movement analytics',
    type: 'object',
    properties: {
      averageBreaks: { type: 'number', example: 4.2 },
      averageMinutes: { type: 'number', example: 28.5 },
      mostActiveDay: { type: 'string', example: 'Tuesday' },
      weeklyTotal: { type: 'number', example: 21 },
      goalAchievementRate: { type: 'number', example: 0.84 }
    }
  })
  movement: {
    averageBreaks: number;
    averageMinutes: number;
    mostActiveDay: string;
    weeklyTotal: number;
    goalAchievementRate: number;
  };

  // Mental wellness analytics
  @ApiProperty({
    description: 'Mental wellness analytics',
    type: 'object',
    properties: {
      averageMoodRating: { type: 'number', example: 4.1 },
      averageStressLevel: { type: 'number', example: 2.3 },
      averageEnergyLevel: { type: 'number', example: 3.8 },
      meditationStreak: { type: 'number', example: 7 },
      totalMindfulnessSessions: { type: 'number', example: 14 }
    }
  })
  mentalWellness: {
    averageMoodRating: number;
    averageStressLevel: number;
    averageEnergyLevel: number;
    meditationStreak: number;
    totalMindfulnessSessions: number;
  };

  // Sleep analytics
  @ApiProperty({
    description: 'Sleep analytics',
    type: 'object',
    properties: {
      averageHours: { type: 'number', example: 7.2 },
      averageQuality: { type: 'number', example: 4.1 },
      consistencyScore: { type: 'number', example: 0.68 },
      bestSleepDay: { type: 'string', example: 'Friday' }
    }
  })
  sleep: {
    averageHours: number;
    averageQuality: number;
    consistencyScore: number;
    bestSleepDay: string;
  };

  // Overall wellness score
  @ApiProperty({
    description: 'Overall wellness analytics',
    type: 'object',
    properties: {
      overallScore: { type: 'number', example: 82 },
      trendDirection: { type: 'string', example: 'upward' },
      streakDays: { type: 'number', example: 12 },
      perfectDaysCount: { type: 'number', example: 3 },
      complianceRate: { type: 'number', example: 0.78 }
    }
  })
  overall: {
    overallScore: number;
    trendDirection: 'upward' | 'stable' | 'downward';
    streakDays: number;
    perfectDaysCount: number;
    complianceRate: number;
  };
}

export class WellnessTrendsDto {
  @ApiProperty({
    description: 'Date of the data point',
    example: '2024-01-15'
  })
  date: string;

  @ApiProperty({
    description: 'Hydration glasses count',
    example: 7
  })
  hydrationGlasses: number;

  @ApiProperty({
    description: 'Movement breaks count',
    example: 5
  })
  movementBreaks: number;

  @ApiProperty({
    description: 'Mood rating',
    example: 4
  })
  moodRating: number;

  @ApiProperty({
    description: 'Stress level',
    example: 2
  })
  stressLevel: number;

  @ApiProperty({
    description: 'Energy level',
    example: 4
  })
  energyLevel: number;

  @ApiProperty({
    description: 'Daily wellness score',
    example: 85
  })
  wellnessScore: number;
}

export class WellnessRecommendationDto {
  @ApiProperty({
    description: 'Recommendation ID',
    example: 'wellness_rec_001'
  })
  id: string;

  @ApiProperty({
    description: 'Type of recommendation',
    example: 'HYDRATION'
  })
  type: string;

  @ApiProperty({
    description: 'Recommendation title',
    example: 'Increase Your Water Intake'
  })
  title: string;

  @ApiProperty({
    description: 'Detailed recommendation description',
    example: "You've been averaging 5 glasses/day. Try setting hourly reminders to reach your goal of 8 glasses!"
  })
  description: string;

  @ApiProperty({
    description: 'Priority level',
    enum: ['LOW', 'MEDIUM', 'HIGH'],
    example: 'MEDIUM'
  })
  priority: 'LOW' | 'MEDIUM' | 'HIGH';

  @ApiProperty({
    description: 'Whether the recommendation is actionable',
    example: true
  })
  actionable: boolean;

  @ApiProperty({
    description: 'Estimated impact',
    example: '+15 wellness score'
  })
  estimatedImpact: string;
}