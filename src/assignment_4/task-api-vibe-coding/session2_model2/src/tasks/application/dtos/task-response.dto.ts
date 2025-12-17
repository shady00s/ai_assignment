import { ApiProperty } from '@nestjs/swagger';
import { Task } from '../../domain/entities/task.entity';
import { TaskStatus } from '../../domain/values/task-status.enum';
import { TaskPriority } from '../../domain/values/task-priority.enum';

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

  static fromDomainEntity(task: Task): TaskResponseDto {
    return {
      id: task.getId().getValue(),
      title: task.getTitle().getValue(),
      description: task.getDescription(),
      status: task.getStatus().getValue(),
      priority: task.getPriority().getValue(),
      createdAt: task.getCreatedAt(),
      updatedAt: task.getUpdatedAt(),
    };
  }
}