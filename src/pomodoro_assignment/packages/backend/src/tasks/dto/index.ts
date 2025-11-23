export { CreateTaskDto } from './create-task.dto';
export { UpdateTaskDto } from './update-task.dto';
export { TaskResponseDto } from './task-response.dto';

// Export enums for frontend compatibility
export const TaskStatus = {
  TODO: 'TODO' as const,
  IN_PROGRESS: 'IN_PROGRESS' as const,
  COMPLETED: 'COMPLETED' as const,
  CANCELLED: 'CANCELLED' as const,
};

export const Priority = {
  LOW: 'LOW' as const,
  MEDIUM: 'MEDIUM' as const,
  HIGH: 'HIGH' as const,
  CRITICAL: 'CRITICAL' as const,
};

export type TaskStatusType = typeof TaskStatus[keyof typeof TaskStatus];
export type PriorityType = typeof Priority[keyof typeof Priority];