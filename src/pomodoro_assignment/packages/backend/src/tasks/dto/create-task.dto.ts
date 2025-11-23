import { IsString, IsOptional, IsEnum, IsInt, IsArray, Min, Max, MaxLength } from 'class-validator';

// String enums for validation - ALIGNED WITH FRONTEND
const Priority = {
  LOW: 'LOW',
  MEDIUM: 'MEDIUM',
  HIGH: 'HIGH',
  URGENT: 'URGENT',
} as const;

const TaskStatus = {
  TODO: 'TODO',
  IN_PROGRESS: 'IN_PROGRESS',
  COMPLETED: 'COMPLETED',
  CANCELLED: 'CANCELLED',
} as const;

type PriorityType = typeof Priority[keyof typeof Priority];
type TaskStatusType = typeof TaskStatus[keyof typeof TaskStatus];

export class CreateTaskDto {
  @IsString()
  @MaxLength(200)
  title: string;

  @IsOptional()
  @IsString()
  @MaxLength(1000)
  description?: string;

  @IsOptional()
  @IsEnum(Object.values(Priority))
  priority?: PriorityType = Priority.MEDIUM;

  @IsOptional()
  dueDate?: Date;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(50)
  estimatedPomodoros?: number = 1;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(10)
  complexity?: number = 1;

  @IsOptional()
  @IsString()
  assigneeId?: string;

  @IsOptional()
  @IsString()
  teamId?: string;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  tags?: string[] = [];
}