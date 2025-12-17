import { TaskId } from '../../domain/values/task-id.value';
import { IQuery } from './query.interface';

export class GetTaskByIdQuery implements IQuery {
  constructor(public readonly id: string) {}

  static create(id: string): GetTaskByIdQuery {
    return new GetTaskByIdQuery(id);
  }

  getTaskId(): TaskId {
    return TaskId.fromString(this.id);
  }
}