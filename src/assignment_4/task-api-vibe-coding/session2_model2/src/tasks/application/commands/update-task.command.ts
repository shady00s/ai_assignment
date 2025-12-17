import { TaskId } from '../../domain/values/task-id.value';
import { TaskTitle } from '../../domain/values/task-title.value';
import { TaskStatusValue } from '../../domain/values/task-status.enum';
import { TaskPriorityValue } from '../../domain/values/task-priority.enum';
import { ICommand } from './command.interface';

export class UpdateTaskCommand implements ICommand {
  constructor(
    public readonly id: string,
    public readonly title?: string,
    public readonly description?: string | null,
    public readonly status?: string,
    public readonly priority?: string
  ) {}

  static create(
    id: string,
    updates: {
      title?: string;
      description?: string | null;
      status?: string;
      priority?: string;
    }
  ): UpdateTaskCommand {
    return new UpdateTaskCommand(
      id,
      updates.title,
      updates.description,
      updates.status,
      updates.priority
    );
  }

  getTaskId(): TaskId {
    return TaskId.fromString(this.id);
  }

  getTitle(): TaskTitle | undefined {
    return this.title ? TaskTitle.create(this.title) : undefined;
  }

  getStatus(): TaskStatusValue | undefined {
    return this.status ? TaskStatusValue.fromString(this.status) : undefined;
  }

  getPriority(): TaskPriorityValue | undefined {
    return this.priority ? TaskPriorityValue.fromString(this.priority) : undefined;
  }
}