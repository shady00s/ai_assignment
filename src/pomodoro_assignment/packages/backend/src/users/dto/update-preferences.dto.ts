import { IsOptional, IsBoolean, IsNumber, IsEnum, Min, Max } from 'class-validator';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

export class UpdatePreferencesDto {
  @ApiPropertyOptional({
    description: 'Work duration in minutes',
    example: 25,
    minimum: 1,
    maximum: 60,
  })
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(60)
  workDuration?: number;

  @ApiPropertyOptional({
    description: 'Short break duration in minutes',
    example: 5,
    minimum: 1,
    maximum: 15,
  })
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(15)
  shortBreakDuration?: number;

  @ApiPropertyOptional({
    description: 'Long break duration in minutes',
    example: 15,
    minimum: 1,
    maximum: 30,
  })
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(30)
  longBreakDuration?: number;

  @ApiPropertyOptional({
    description: 'Long break interval (number of work sessions)',
    example: 4,
    minimum: 2,
    maximum: 10,
  })
  @IsOptional()
  @IsNumber()
  @Min(2)
  @Max(10)
  longBreakInterval?: number;

  @ApiPropertyOptional({
    description: 'Auto-start breaks',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  autoStartBreaks?: boolean;

  @ApiPropertyOptional({
    description: 'Auto-start work sessions',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  autoStartWork?: boolean;

  @ApiPropertyOptional({
    description: 'Sound enabled',
    example: true,
  })
  @IsOptional()
  @IsBoolean()
  soundEnabled?: boolean;

  @ApiPropertyOptional({
    description: 'Volume level (0-100)',
    example: 70,
    minimum: 0,
    maximum: 100,
  })
  @IsOptional()
  @IsNumber()
  @Min(0)
  @Max(100)
  volume?: number;

  @ApiPropertyOptional({
    description: 'Ambient sound type',
    enum: ['forest', 'ocean', 'cafe', 'rain', 'none'],
    example: 'forest',
  })
  @IsOptional()
  @IsEnum(['forest', 'ocean', 'cafe', 'rain', 'none'])
  ambientSound?: string;

  @ApiPropertyOptional({
    description: 'Dark mode enabled',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  darkMode?: boolean;
}