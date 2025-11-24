import { ApiProperty } from '@nestjs/swagger';
import { IsNumber, IsOptional, IsInt, Min, Max } from 'class-validator';
import { Transform } from 'class-transformer';

export class CreateWellnessEntryDto {
  @ApiProperty({
    description: 'Date for the wellness entry (defaults to today)',
    example: '2024-01-15',
    required: false
  })
  @IsOptional()
  @Transform(({ value }) => value ? new Date(value) : new Date())
  date?: Date;

  // Hydration tracking
  @ApiProperty({
    description: 'Number of glasses of water consumed',
    example: 6,
    minimum: 0,
    maximum: 20
  })
  @IsNumber()
  @Min(0)
  @Max(20)
  hydrationGlasses: number;

  @ApiProperty({
    description: 'Daily hydration goal in glasses',
    example: 8,
    minimum: 1,
    maximum: 20
  })
  @IsNumber()
  @Min(1)
  @Max(20)
  hydrationGoal: number;

  // Movement tracking
  @ApiProperty({
    description: 'Number of movement breaks taken',
    example: 4,
    minimum: 0
  })
  @IsNumber()
  @Min(0)
  movementBreaks: number;

  @ApiProperty({
    description: 'Total movement minutes',
    example: 25,
    minimum: 0
  })
  @IsNumber()
  @Min(0)
  movementMinutes: number;

  @ApiProperty({
    description: 'Step count (optional)',
    example: 8500,
    minimum: 0,
    required: false
  })
  @IsOptional()
  @IsNumber()
  @Min(0)
  stepsCount?: number;

  // Mental wellness
  @ApiProperty({
    description: 'Time spent meditating in minutes',
    example: 15,
    minimum: 0
  })
  @IsNumber()
  @Min(0)
  meditationMinutes: number;

  @ApiProperty({
    description: 'Number of breathing exercises completed',
    example: 3,
    minimum: 0
  })
  @IsNumber()
  @Min(0)
  breathingExercises: number;

  @ApiProperty({
    description: 'Number of mindfulness sessions',
    example: 2,
    minimum: 0
  })
  @IsNumber()
  @Min(0)
  mindfulnessSessions: number;

  // Self-reported metrics
  @ApiProperty({
    description: 'Mood rating (1=very poor, 5=excellent)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  moodRating: number;

  @ApiProperty({
    description: 'Stress level (1=very low, 5=very high)',
    example: 2,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  stressLevel: number;

  @ApiProperty({
    description: 'Energy level (1=very low, 5=very high)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  energyLevel: number;

  @ApiProperty({
    description: 'Sleep quality rating (1=very poor, 5=excellent)',
    example: 4,
    minimum: 1,
    maximum: 5,
    required: false
  })
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(5)
  sleepQuality?: number;

  @ApiProperty({
    description: 'Hours of sleep',
    example: 7.5,
    minimum: 0,
    maximum: 24,
    required: false
  })
  @IsOptional()
  @IsNumber()
  @Min(0)
  @Max(24)
  sleepHours?: number;

  // Session-based wellness
  @ApiProperty({
    description: 'Number of posture checks completed',
    example: 6,
    minimum: 0
  })
  @IsNumber()
  @Min(0)
  postureChecks: number;

  @ApiProperty({
    description: 'Number of eye rest breaks taken',
    example: 4,
    minimum: 0
  })
  @IsNumber()
  @Min(0)
  eyeRestBreaks: number;
}