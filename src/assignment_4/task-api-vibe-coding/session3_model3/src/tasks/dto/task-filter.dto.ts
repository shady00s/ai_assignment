import { TaskPriority, TaskStatus } from '@prisma/client';

export class TaskFilterDto {
  status?: TaskStatus;
  priority?: TaskPriority;
}