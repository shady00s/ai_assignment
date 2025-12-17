import { ApiPropertyOptional } from '@nestjs/swagger';

export class TaskFilterDto {
  @ApiPropertyOptional({
    description: 'Filter tasks by status',
    enum: ['PENDING', 'IN_PROGRESS', 'COMPLETED'],
    example: 'PENDING',
  })
  status?: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED';

  @ApiPropertyOptional({
    description: 'Filter tasks by priority level',
    enum: ['LOW', 'MEDIUM', 'HIGH'],
    example: 'HIGH',
  })
  priority?: 'LOW' | 'MEDIUM' | 'HIGH';
}