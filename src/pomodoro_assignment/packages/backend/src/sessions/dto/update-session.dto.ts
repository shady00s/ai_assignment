import { IsString, IsOptional, IsEnum, IsInt, Min, Max } from 'class-validator';

// String enums for validation
const SessionType = {
  POMODORO: 'POMODORO',
  SHORT_BREAK: 'SHORT_BREAK',
  LONG_BREAK: 'LONG_BREAK',
  CUSTOM: 'CUSTOM',
} as const;

type SessionTypeType = typeof SessionType[keyof typeof SessionType];

export class UpdateSessionDto {
  @IsOptional()
  @IsString()
  taskId?: string;

  @IsOptional()
  @IsEnum(Object.values(SessionType))
  type?: SessionTypeType;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(180)
  duration?: number;

  @IsOptional()
  @IsString()
  notes?: string;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(5)
  quality?: number;

  @IsOptional()
  @IsInt()
  @Min(0)
  interruptions?: number;
}