import { ApiProperty } from '@nestjs/swagger';
import { IsDateString, IsNumber, IsOptional, IsString, Min, Max } from 'class-validator';
import { Transform } from 'class-transformer';

export class WellnessHistoryQueryDto {
  @ApiProperty({
    description: 'Start date for history query',
    example: '2024-01-01',
    required: false
  })
  @IsOptional()
  @IsDateString()
  startDate?: string;

  @ApiProperty({
    description: 'End date for history query',
    example: '2024-01-31',
    required: false
  })
  @IsOptional()
  @IsDateString()
  endDate?: string;

  @ApiProperty({
    description: 'Number of days to look back from today',
    example: 30,
    minimum: 1,
    maximum: 365,
    required: false
  })
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(365)
  days?: number;

  @ApiProperty({
    description: 'Page number for pagination',
    example: 1,
    minimum: 1,
    required: false
  })
  @IsOptional()
  @Transform(({ value }) => parseInt(value))
  @IsNumber()
  @Min(1)
  page?: number = 1;

  @ApiProperty({
    description: 'Number of items per page',
    example: 10,
    minimum: 1,
    maximum: 100,
    required: false
  })
  @IsOptional()
  @Transform(({ value }) => parseInt(value))
  @IsNumber()
  @Min(1)
  @Max(100)
  limit?: number = 10;

  @ApiProperty({
    description: 'Sort field',
    example: 'date',
    required: false
  })
  @IsOptional()
  @IsString()
  sortBy?: string = 'date';

  @ApiProperty({
    description: 'Sort order',
    example: 'desc',
    required: false
  })
  @IsOptional()
  @IsString()
  sortOrder?: 'asc' | 'desc' = 'desc';
}

export class WellnessAnalyticsQueryDto {
  @ApiProperty({
    description: 'Analysis period in days',
    example: 30,
    minimum: 1,
    maximum: 365,
    required: false
  })
  @IsOptional()
  @Transform(({ value }) => parseInt(value))
  @IsNumber()
  @Min(1)
  @Max(365)
  days?: number = 30;

  @ApiProperty({
    description: 'Start date for analytics',
    example: '2024-01-01',
    required: false
  })
  @IsOptional()
  @IsDateString()
  startDate?: string;

  @ApiProperty({
    description: 'End date for analytics',
    example: '2024-01-31',
    required: false
  })
  @IsOptional()
  @IsDateString()
  endDate?: string;

  @ApiProperty({
    description: 'Category to focus analysis on',
    example: 'HYDRATION',
    required: false
  })
  @IsOptional()
  @IsString()
  category?: 'HYDRATION' | 'MOVEMENT' | 'MEDITATION' | 'SLEEP' | 'ALL';

  @ApiProperty({
    description: 'Include recommendations in response',
    example: true,
    required: false
  })
  @IsOptional()
  @Transform(({ value }) => value === 'true' || value === true)
  includeRecommendations?: boolean = true;

  @ApiProperty({
    description: 'Include trends data in response',
    example: true,
    required: false
  })
  @IsOptional()
  @Transform(({ value }) => value === 'true' || value === true)
  includeTrends?: boolean = false;
}