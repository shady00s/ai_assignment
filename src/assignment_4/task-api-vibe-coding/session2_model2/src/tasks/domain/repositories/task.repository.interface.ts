import { Task } from '../entities/task.entity';
import { TaskId } from '../values/task-id.value';
import { TaskStatusValue } from '../values/task-status.enum';
import { TaskPriorityValue } from '../values/task-priority.enum';

export interface TaskRepository {
  save(task: Task): Promise<void>;
  findById(id: TaskId): Promise<Task | null>;
  findAll(status?: TaskStatusValue, priority?: TaskPriorityValue): Promise<Task[]>;
  delete(id: TaskId): Promise<void>;
  exists(id: TaskId): Promise<boolean>;
}