import { TaskTitle } from '../../domain/values/task-title.value';
import { TaskStatusValue } from '../../domain/values/task-status.enum';
import { TaskPriorityValue } from '../../domain/values/task-priority.enum';
import { ICommand } from './command.interface';

export class CreateTaskCommand implements ICommand {
  constructor(
    public readonly title: string,
    public readonly description: string | null,
    public readonly priority: string
  ) {}

  static create(title: string, description?: string, priority?: string): CreateTaskCommand {
    const taskTitle = TaskTitle.create(title);
    const taskPriority = priority
      ? TaskPriorityValue.fromString(priority)
      : TaskPriorityValue.default();

    return new CreateTaskCommand(
      taskTitle.getValue(),
      description || null,
      taskPriority.getValue()
    );
  }

  getTitle(): TaskTitle {
    return TaskTitle.create(this.title);
  }

  getDescription(): string | null {
    return this.description;
  }

  getPriority(): TaskPriorityValue {
    return TaskPriorityValue.fromString(this.priority);
  }
}