import { ApiProperty } from '@nestjs/swagger';
import { IsNumber, IsOptional, Min, Max, IsString } from 'class-validator';

export class UpdateMoodDto {
  @ApiProperty({
    description: 'Mood rating (1=very poor, 5=excellent)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  mood: number;

  @ApiProperty({
    description: 'Stress level (1=very low, 5=very high)',
    example: 2,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  stress: number;

  @ApiProperty({
    description: 'Energy level (1=very low, 5=very high)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  energy: number;
}

export class LogSleepDto {
  @ApiProperty({
    description: 'Hours of sleep',
    example: 7.5,
    minimum: 0,
    maximum: 24
  })
  @IsNumber()
  @Min(0)
  @Max(24)
  hours: number;

  @ApiProperty({
    description: 'Sleep quality rating (1=very poor, 5=excellent)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  quality: number;

  @ApiProperty({
    description: 'Date for the sleep entry (defaults to today)',
    example: '2024-01-15',
    required: false
  })
  @IsOptional()
  @IsString()
  date?: string;
}

export class QuickMoodDto {
  @ApiProperty({
    description: 'Mood rating (1=very poor, 5=excellent)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  mood: number;

  @ApiProperty({
    description: 'Energy level (1=very low, 5=very high)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  energy: number;

  @ApiProperty({
    description: 'Stress level (1=very low, 5=very high)',
    example: 2,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  stress: number;
}