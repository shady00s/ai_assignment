import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsNumber, IsBoolean, IsEnum, Min, Max } from 'class-validator';

export enum WellnessGoalCategory {
  HYDRATION = 'HYDRATION',
  MOVEMENT = 'MOVEMENT',
  MEDITATION = 'MEDITATION',
  SLEEP = 'SLEEP'
}

export enum WellnessGoalPeriod {
  DAILY = 'DAILY',
  WEEKLY = 'WEEKLY',
  MONTHLY = 'MONTHLY'
}

export class CreateWellnessGoalDto {
  @ApiProperty({
    description: 'Category of wellness goal',
    enum: WellnessGoalCategory,
    example: WellnessGoalCategory.HYDRATION
  })
  @IsEnum(WellnessGoalCategory)
  category: WellnessGoalCategory;

  @ApiProperty({
    description: 'Target value for the goal (e.g., 8 glasses, 10000 steps)',
    example: 8,
    minimum: 1
  })
  @IsNumber()
  @Min(1)
  targetValue: number;

  @ApiProperty({
    description: 'Period for the goal',
    enum: WellnessGoalPeriod,
    example: WellnessGoalPeriod.DAILY
  })
  @IsEnum(WellnessGoalPeriod)
  period: WellnessGoalPeriod;

  @ApiProperty({
    description: 'Whether the goal is currently active',
    example: true,
    default: true
  })
  @IsBoolean()
  active: boolean;
}

export class UpdateWellnessGoalDto {
  @ApiProperty({
    description: 'Category of wellness goal',
    enum: WellnessGoalCategory,
    example: WellnessGoalCategory.HYDRATION,
    required: false
  })
  @IsEnum(WellnessGoalCategory)
  @IsString()
  category?: WellnessGoalCategory;

  @ApiProperty({
    description: 'Target value for the goal',
    example: 8,
    minimum: 1,
    required: false
  })
  @IsNumber()
  @Min(1)
  targetValue?: number;

  @ApiProperty({
    description: 'Period for the goal',
    enum: WellnessGoalPeriod,
    example: WellnessGoalPeriod.DAILY,
    required: false
  })
  @IsEnum(WellnessGoalPeriod)
  period?: WellnessGoalPeriod;

  @ApiProperty({
    description: 'Whether the goal is currently active',
    example: true,
    required: false
  })
  @IsBoolean()
  active?: boolean;
}

export class WellnessGoalResponseDto {
  @ApiProperty({
    description: 'Goal ID',
    example: 'clym8d1230000sbdp1234567'
  })
  id: string;

  @ApiProperty({
    description: 'User ID',
    example: 'clym8d1230000sbdp1234567'
  })
  userId: string;

  @ApiProperty({
    description: 'Category of wellness goal',
    enum: WellnessGoalCategory,
    example: WellnessGoalCategory.HYDRATION
  })
  category: WellnessGoalCategory;

  @ApiProperty({
    description: 'Target value for the goal',
    example: 8
  })
  targetValue: number;

  @ApiProperty({
    description: 'Period for the goal',
    enum: WellnessGoalPeriod,
    example: WellnessGoalPeriod.DAILY
  })
  period: WellnessGoalPeriod;

  @ApiProperty({
    description: 'Whether the goal is currently active',
    example: true
  })
  active: boolean;

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
    description: 'Current progress towards the goal',
    example: 6,
    required: false
  })
  currentProgress?: number;

  @ApiProperty({
    description: 'Progress percentage',
    example: 75,
    required: false
  })
  progressPercentage?: number;
}