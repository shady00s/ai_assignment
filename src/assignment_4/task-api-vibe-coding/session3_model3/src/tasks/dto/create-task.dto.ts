import { TaskPriority, TaskStatus } from '@prisma/client';

export class CreateTaskDto {
  title: string;
  description?: string;
  priority?: TaskPriority;
}