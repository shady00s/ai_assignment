import { ApiProperty } from '@nestjs/swagger';
import { IsNumber, IsOptional, IsBoolean, Min, Max } from 'class-validator';

export class LogPostureCheckDto {
  @ApiProperty({
    description: 'Whether the posture check was completed',
    example: true
  })
  @IsBoolean()
  completed: boolean;
}

export class LogEyeRestDto {
  @ApiProperty({
    description: 'Duration of eye rest break in minutes',
    example: 2,
    minimum: 1,
    maximum: 30
  })
  @IsNumber()
  @Min(1)
  @Max(30)
  duration: number;

  @ApiProperty({
    description: 'Whether the eye rest break was completed successfully',
    example: true
  })
  @IsBoolean()
  completed: boolean;
}