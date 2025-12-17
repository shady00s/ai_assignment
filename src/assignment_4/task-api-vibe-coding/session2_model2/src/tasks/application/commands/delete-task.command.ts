import { TaskId } from '../../domain/values/task-id.value';
import { ICommand } from './command.interface';

export class DeleteTaskCommand implements ICommand {
  constructor(public readonly id: string) {}

  static create(id: string): DeleteTaskCommand {
    return new DeleteTaskCommand(id);
  }

  getTaskId(): TaskId {
    return TaskId.fromString(this.id);
  }
}