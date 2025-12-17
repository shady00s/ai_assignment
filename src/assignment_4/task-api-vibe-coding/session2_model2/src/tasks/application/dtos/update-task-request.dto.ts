import { ApiPropertyOptional } from '@nestjs/swagger';

export class UpdateTaskRequestDto {
  @ApiPropertyOptional({
    description: 'The updated title of the task',
    example: 'Updated task title',
    minLength: 1,
    maxLength: 255,
  })
  title?: string;

  @ApiPropertyOptional({
    description: 'The updated description of the task',
    example: 'Updated task description with more details about CQRS implementation',
    maxLength: 1000,
    nullable: true,
  })
  description?: string | null;

  @ApiPropertyOptional({
    description: 'The updated status of the task',
    enum: ['PENDING', 'IN_PROGRESS', 'COMPLETED'],
    example: 'IN_PROGRESS',
  })
  status?: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED';

  @ApiPropertyOptional({
    description: 'The updated priority level of the task',
    enum: ['LOW', 'MEDIUM', 'HIGH'],
    example: 'HIGH',
  })
  priority?: 'LOW' | 'MEDIUM' | 'HIGH';
}