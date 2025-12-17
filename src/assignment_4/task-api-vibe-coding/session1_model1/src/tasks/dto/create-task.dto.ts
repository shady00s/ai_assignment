import { TaskPriority, TaskStatus } from '@prisma/client';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

export class CreateTaskDto {
  @ApiProperty({
    description: 'The title of the task',
    example: 'Complete project documentation',
    minLength: 1,
    maxLength: 255,
  })
  title: string;

  @ApiPropertyOptional({
    description: 'A detailed description of the task',
    example: 'Write comprehensive documentation for the Task Management API including endpoints, request/response schemas, and usage examples',
    maxLength: 1000,
  })
  description?: string;

  @ApiPropertyOptional({
    description: 'The priority level of the task',
    enum: TaskPriority,
    example: TaskPriority.MEDIUM,
    default: TaskPriority.MEDIUM,
  })
  priority?: TaskPriority;
}