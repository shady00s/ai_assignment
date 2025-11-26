import { ApiProperty } from '@nestjs/swagger';
import { IsNumber, IsOptional, Min, Max } from 'class-validator';

export class IncrementHydrationDto {
  @ApiProperty({
    description: 'Number of glasses of water to increment',
    example: 1,
    minimum: 1,
    maximum: 20
  })
  @IsNumber()
  @Min(1)
  @Max(20)
  glasses: number;
}

export class SetHydrationGoalDto {
  @ApiProperty({
    description: 'Daily hydration goal in glasses',
    example: 8,
    minimum: 1,
    maximum: 20
  })
  @IsNumber()
  @Min(1)
  @Max(20)
  goal: number;
}

export class QuickWaterDto {
  @ApiProperty({
    description: 'Number of glasses of water consumed (quick log)',
    example: 1,
    minimum: 1,
    maximum: 20
  })
  @IsNumber()
  @Min(1)
  @Max(20)
  glasses: number;
}