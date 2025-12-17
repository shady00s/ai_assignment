import { Task, TaskPriority, TaskStatus } from '@prisma/client';
import { ApiProperty } from '@nestjs/swagger';

export class TaskResponseDto {
  @ApiProperty({
    description: 'Unique identifier for the task',
    example: 'cuy1t2y3k0000l2z8s0r1x2a4',
  })
  id: string;

  @ApiProperty({
    description: 'The title of the task',
    example: 'Complete project documentation',
  })
  title: string;

  @ApiProperty({
    description: 'A detailed description of the task',
    example: 'Write comprehensive documentation for the Task Management API',
    nullable: true,
  })
  description: string | null;

  @ApiProperty({
    description: 'The current status of the task',
    enum: TaskStatus,
    example: TaskStatus.IN_PROGRESS,
  })
  status: TaskStatus;

  @ApiProperty({
    description: 'The priority level of the task',
    enum: TaskPriority,
    example: TaskPriority.HIGH,
  })
  priority: TaskPriority;

  @ApiProperty({
    description: 'Timestamp when the task was created',
    example: '2024-01-15T10:30:00.000Z',
  })
  createdAt: Date;

  @ApiProperty({
    description: 'Timestamp when the task was last updated',
    example: '2024-01-15T14:20:00.000Z',
  })
  updatedAt: Date;

  static fromEntity(task: Task): TaskResponseDto {
    return {
      id: task.id,
      title: task.title,
      description: task.description,
      status: task.status,
      priority: task.priority,
      createdAt: task.createdAt,
      updatedAt: task.updatedAt,
    };
  }
}