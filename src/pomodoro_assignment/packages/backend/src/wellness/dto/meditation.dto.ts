import { ApiProperty } from '@nestjs/swagger';
import { IsNumber, IsString, IsOptional, Min, Max } from 'class-validator';

export class LogMeditationDto {
  @ApiProperty({
    description: 'Duration of meditation in minutes',
    example: 15,
    minimum: 1,
    maximum: 180
  })
  @IsNumber()
  @Min(1)
  @Max(180)
  minutes: number;

  @ApiProperty({
    description: 'Type of meditation session',
    example: 'mindfulness',
    required: false
  })
  @IsOptional()
  @IsString()
  type?: string;

  @ApiProperty({
    description: 'Quality rating of meditation (1=very poor, 5=excellent)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  quality: number;

  @ApiProperty({
    description: 'Optional notes about the meditation session',
    example: 'Felt very focused and calm',
    required: false
  })
  @IsOptional()
  @IsString()
  notes?: string;
}

export class CompleteMeditationDto {
  @ApiProperty({
    description: 'ID of the meditation session to complete',
    example: 'clym8d1230000sbdp1234567'
  })
  @IsString()
  sessionId: string;

  @ApiProperty({
    description: 'Quality rating of meditation (1=very poor, 5=excellent)',
    example: 4,
    minimum: 1,
    maximum: 5
  })
  @IsNumber()
  @Min(1)
  @Max(5)
  quality: number;

  @ApiProperty({
    description: 'Optional notes about the completed session',
    example: 'Great session, felt very relaxed',
    required: false
  })
  @IsOptional()
  @IsString()
  notes?: string;
}

export class LogBreathingDto {
  @ApiProperty({
    description: 'Duration of breathing exercise in minutes',
    example: 5,
    minimum: 1,
    maximum: 60
  })
  @IsNumber()
  @Min(1)
  @Max(60)
  duration: number;

  @ApiProperty({
    description: 'Type of breathing exercise',
    example: '4-7-8 breathing',
    required: false
  })
  @IsOptional()
  @IsString()
  type?: string;

  @ApiProperty({
    description: 'Number of rounds completed',
    example: 5,
    minimum: 1,
    maximum: 50,
    required: false
  })
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(50)
  rounds?: number;
}

export class QuickMeditationDto {
  @ApiProperty({
    description: 'Minutes of meditation for quick logging',
    example: 10,
    minimum: 1,
    maximum: 180
  })
  @IsNumber()
  @Min(1)
  @Max(180)
  minutes: number;
}