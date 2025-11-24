import { ApiProperty } from '@nestjs/swagger';

export class FocusAnalyticsDto {
  @ApiProperty({
    description: 'Minutes of focus time today',
    example: 225,
    minimum: 0
  })
  dailyFocusTime: number;

  @ApiProperty({
    description: 'Minutes of focus time this week',
    example: 1575,
    minimum: 0
  })
  weeklyFocusTime: number;

  @ApiProperty({
    description: 'Minutes of focus time this month',
    example: 6750,
    minimum: 0
  })
  monthlyFocusTime: number;

  @ApiProperty({
    description: 'Average session length in minutes',
    example: 25.5,
    minimum: 1
  })
  averageSessionLength: number;

  @ApiProperty({
    description: 'Hours with most focus time (24-hour format)',
    example: [9, 10, 14],
    type: [Number],
    minItems: 1,
    maxItems: 5
  })
  peakFocusHours: number[];

  @ApiProperty({
    description: 'Focus trend direction compared to previous period',
    enum: ['IMPROVING', 'DECLINING', 'STABLE'],
    example: 'IMPROVING'
  })
  focusTrend: 'IMPROVING' | 'DECLINING' | 'STABLE';

  @ApiProperty({
    description: 'Session completion rate percentage',
    example: 85.5,
    minimum: 0,
    maximum: 100
  })
  completionRate: number;
}