import { ApiProperty } from '@nestjs/swagger';
import { IsNumber, IsString, IsOptional, IsEnum, Min, Max } from 'class-validator';

export enum MovementIntensity {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH'
}

export class LogMovementDto {
  @ApiProperty({
    description: 'Duration of movement break in minutes',
    example: 5,
    minimum: 1,
    maximum: 1440
  })
  @IsNumber()
  @Min(1)
  @Max(1440)
  duration: number;

  @ApiProperty({
    description: 'Type of movement activity',
    example: 'walking',
    required: false
  })
  @IsOptional()
  @IsString()
  type?: string;

  @ApiProperty({
    description: 'Intensity level of the movement',
    enum: MovementIntensity,
    example: MovementIntensity.MEDIUM,
    required: false
  })
  @IsOptional()
  @IsEnum(MovementIntensity)
  intensity?: MovementIntensity;
}

export class LogStepsDto {
  @ApiProperty({
    description: 'Number of steps taken',
    example: 8500,
    minimum: 0,
    maximum: 100000
  })
  @IsNumber()
  @Min(0)
  @Max(100000)
  steps: number;

  @ApiProperty({
    description: 'Date for the step count (defaults to today)',
    example: '2024-01-15',
    required: false
  })
  @IsOptional()
  @IsString()
  date?: string;
}

export class SetMovementGoalDto {
  @ApiProperty({
    description: 'Daily goal for movement breaks',
    example: 5,
    minimum: 1,
    maximum: 50
  })
  @IsNumber()
  @Min(1)
  @Max(50)
  dailyBreaks: number;

  @ApiProperty({
    description: 'Daily goal for movement minutes',
    example: 30,
    minimum: 1,
    maximum: 1440
  })
  @IsNumber()
  @Min(1)
  @Max(1440)
  dailyMinutes: number;
}

export class QuickMovementDto {
  @ApiProperty({
    description: 'Minutes of movement for quick logging',
    example: 5,
    minimum: 1,
    maximum: 1440
  })
  @IsNumber()
  @Min(1)
  @Max(1440)
  minutes: number;
}