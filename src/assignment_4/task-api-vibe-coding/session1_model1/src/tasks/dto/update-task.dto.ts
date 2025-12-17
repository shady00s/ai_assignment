import { TaskPriority, TaskStatus } from '@prisma/client';
import { ApiPropertyOptional } from '@nestjs/swagger';

export class UpdateTaskDto {
  @ApiPropertyOptional({
    description: 'The updated title of the task',
    example: 'Updated task title',
    minLength: 1,
    maxLength: 255,
  })
  title?: string;

  @ApiPropertyOptional({
    description: 'The updated description of the task',
    example: 'Updated task description with more details',
    maxLength: 1000,
  })
  description?: string;

  @ApiPropertyOptional({
    description: 'The updated status of the task',
    enum: TaskStatus,
    example: TaskStatus.IN_PROGRESS,
  })
  status?: TaskStatus;

  @ApiPropertyOptional({
    description: 'The updated priority level of the task',
    enum: TaskPriority,
    example: TaskPriority.HIGH,
  })
  priority?: TaskPriority;
}