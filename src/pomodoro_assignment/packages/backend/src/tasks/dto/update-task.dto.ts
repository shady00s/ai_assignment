import { IsString, IsOptional, IsEnum, IsInt, IsArray, IsDateString, Min, Max, MaxLength } from 'class-validator';

// String enums for validation - ALIGNED WITH FRONTEND
const TaskStatus = {
  TODO: 'TODO',
  IN_PROGRESS: 'IN_PROGRESS',
  COMPLETED: 'COMPLETED',
  CANCELLED: 'CANCELLED',
} as const;

const Priority = {
  LOW: 'LOW',
  MEDIUM: 'MEDIUM',
  HIGH: 'HIGH',
  URGENT: 'URGENT',
} as const;

type TaskStatusType = typeof TaskStatus[keyof typeof TaskStatus];
type PriorityType = typeof Priority[keyof typeof Priority];

export class UpdateTaskDto {
  @IsOptional()
  @IsString()
  @MaxLength(200)
  title?: string;

  @IsOptional()
  @IsString()
  @MaxLength(1000)
  description?: string;

  @IsOptional()
  @IsEnum(Object.values(TaskStatus))
  status?: TaskStatusType;

  @IsOptional()
  @IsEnum(Object.values(Priority))
  priority?: PriorityType;

  @IsOptional()
  @IsDateString()
  dueDate?: string;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(50)
  estimatedPomodoros?: number;

  @IsOptional()
  @IsInt()
  @Min(0)
  completedPomodoros?: number;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(10)
  complexity?: number;

  @IsOptional()
  @IsString()
  assigneeId?: string;

  @IsOptional()
  @IsString()
  teamId?: string;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  tags?: string[];
}