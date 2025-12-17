import { TaskStatusValue } from '../../domain/values/task-status.enum';
import { TaskPriorityValue } from '../../domain/values/task-priority.enum';
import { IQuery } from './query.interface';

export class GetTasksQuery implements IQuery {
  constructor(
    public readonly status?: string,
    public readonly priority?: string
  ) {}

  static create(filters?: { status?: string; priority?: string }): GetTasksQuery {
    return new GetTasksQuery(filters?.status, filters?.priority);
  }

  getStatus(): TaskStatusValue | undefined {
    return this.status ? TaskStatusValue.fromString(this.status) : undefined;
  }

  getPriority(): TaskPriorityValue | undefined {
    return this.priority ? TaskPriorityValue.fromString(this.priority) : undefined;
  }
}