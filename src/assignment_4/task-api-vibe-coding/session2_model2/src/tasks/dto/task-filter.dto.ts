import { TaskPriority, TaskStatus } from '@prisma/client';
import { ApiPropertyOptional } from '@nestjs/swagger';

export class TaskFilterDto {
  @ApiPropertyOptional({
    description: 'Filter tasks by status',
    enum: TaskStatus,
    example: TaskStatus.PENDING,
  })
  status?: TaskStatus;

  @ApiPropertyOptional({
    description: 'Filter tasks by priority level',
    enum: TaskPriority,
    example: TaskPriority.HIGH,
  })
  priority?: TaskPriority;
}