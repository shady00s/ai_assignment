import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

export class CreateTaskRequestDto {
  @ApiProperty({
    description: 'The title of the task',
    example: 'Complete project documentation',
    minLength: 1,
    maxLength: 255,
  })
  title: string;

  @ApiPropertyOptional({
    description: 'A detailed description of the task',
    example: 'Write comprehensive documentation for the Task Management API including CQRS patterns',
    maxLength: 1000,
    nullable: true,
  })
  description?: string | null;

  @ApiPropertyOptional({
    description: 'The priority level of the task',
    enum: ['LOW', 'MEDIUM', 'HIGH'],
    example: 'MEDIUM',
    default: 'MEDIUM',
  })
  priority?: 'LOW' | 'MEDIUM' | 'HIGH';
}