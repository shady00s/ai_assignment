import { IsOptional, IsISO8601 } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class AnalyticsQueryDto {
  @ApiProperty({
    description: 'Start date for analytics range (ISO 8601)',
    required: false,
    example: '2024-01-01T00:00:00.000Z'
  })
  @IsOptional()
  @IsISO8601()
  startDate?: string;

  @ApiProperty({
    description: 'End date for analytics range (ISO 8601)',
    required: false,
    example: '2024-01-31T23:59:59.999Z'
  })
  @IsOptional()
  @IsISO8601()
  endDate?: string;
}