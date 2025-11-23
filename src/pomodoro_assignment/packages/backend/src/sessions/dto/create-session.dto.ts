import { IsString, IsOptional, IsEnum, IsInt, Min, Max } from 'class-validator';

// String enums for validation
const SessionType = {
  POMODORO: 'POMODORO',
  SHORT_BREAK: 'SHORT_BREAK',
  LONG_BREAK: 'LONG_BREAK',
  CUSTOM: 'CUSTOM',
} as const;

type SessionTypeType = typeof SessionType[keyof typeof SessionType];

export class CreateSessionDto {
  @IsOptional()
  @IsString()
  taskId?: string;

  @IsOptional()
  @IsEnum(Object.values(SessionType))
  type?: SessionTypeType = SessionType.POMODORO;

  @IsInt()
  @Min(1)
  @Max(180) // Max 3 hours
  duration: number;

  @IsOptional()
  @IsString()
  notes?: string;
}