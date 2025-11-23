import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

export class TaskResponseDto {
  @ApiProperty({
    description: 'Task ID',
    example: 'task-123',
  })
  id: string;

  @ApiProperty({
    description: 'Task title',
    example: 'Complete project documentation',
  })
  title: string;

  @ApiPropertyOptional({
    description: 'Task description',
    example: 'Write comprehensive documentation for the new API endpoints',
  })
  description?: string;

  @ApiProperty({
    description: 'Task status',
    enum: ['TODO', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED'],
    example: 'IN_PROGRESS',
  })
  status: string;

  @ApiProperty({
    description: 'Task priority',
    enum: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
    example: 'HIGH',
  })
  priority: string;

  @ApiProperty({
    description: 'Estimated number of pomodoros',
    example: 3,
  })
  estimatedPomodoros: number;

  @ApiProperty({
    description: 'Completed pomodoros',
    example: 1,
  })
  completedPomodoros: number;

  @ApiPropertyOptional({
    description: 'ID of the user assigned to this task',
    example: 'user-456',
  })
  assigneeId?: string;

  @ApiPropertyOptional({
    description: 'User assigned to this task',
    type: 'object',
    properties: {
      id: { type: 'string', example: 'user-456' },
      email: { type: 'string', example: 'user@example.com' },
      firstName: { type: 'string', example: 'John' },
      lastName: { type: 'string', example: 'Doe' },
      avatar: { type: 'string', example: 'https://example.com/avatar.jpg' },
    },
  })
  assignee?: any;

  @ApiPropertyOptional({
    description: 'Task due date (ISO string)',
    example: '2024-01-15T10:00:00.000Z',
  })
  dueDate?: string;

  @ApiProperty({
    description: 'Task creation date (ISO string)',
    example: '2024-01-01T09:00:00.000Z',
  })
  createdAt: string;

  @ApiProperty({
    description: 'Task last update date (ISO string)',
    example: '2024-01-10T14:30:00.000Z',
  })
  updatedAt: string;

  @ApiPropertyOptional({
    description: 'Task completion date (ISO string)',
    example: '2024-01-12T16:00:00.000Z',
  })
  completedAt?: string;

  @ApiProperty({
    description: 'Task tags',
    type: 'array',
    items: { type: 'string' },
    example: ['urgent', 'frontend', 'documentation'],
  })
  tags: string[];
}