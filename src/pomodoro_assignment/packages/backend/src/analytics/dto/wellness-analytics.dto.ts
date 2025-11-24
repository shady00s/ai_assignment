import { ApiProperty } from '@nestjs/swagger';

export class WellnessAnalyticsDto {
  @ApiProperty({
    description: 'Minutes of mindfulness/break activities',
    example: 45,
    minimum: 0
  })
  mindfulnessMinutes: number;

  @ApiProperty({
    description: 'Daily hydration goal in glasses of water',
    example: 8,
    minimum: 1,
    maximum: 15
  })
  hydrationGoal: number;

  @ApiProperty({
    description: 'Current hydration intake for today',
    example: 6,
    minimum: 0,
    maximum: 15
  })
  hydrationCurrent: number;

  @ApiProperty({
    description: 'Daily movement/break goal',
    example: 5,
    minimum: 1,
    maximum: 10
  })
  movementGoal: number;

  @ApiProperty({
    description: 'Current movement breaks taken today',
    example: 3,
    minimum: 0,
    maximum: 10
  })
  movementCurrent: number;

  @ApiProperty({
    description: 'Current mood rating (1=very poor, 5=excellent)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  moodRating: number;

  @ApiProperty({
    description: 'Current stress level (1=very low, 5=very high)',
    example: 2,
    minimum: 1,
    maximum: 5
  })
  stressLevel: number;

  @ApiProperty({
    description: 'Current energy level (1=very low, 5=very high)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  energyLevel: number;
}