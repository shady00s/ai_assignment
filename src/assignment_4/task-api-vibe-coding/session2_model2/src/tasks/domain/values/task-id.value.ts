import { v4 as uuidv4 } from 'uuid';
import { ValueObject } from './base.value-object';

export class TaskId extends ValueObject<string> {
  private constructor(value: string) {
    super(value);
    this.validate();
  }

  static generate(): TaskId {
    return new TaskId(uuidv4());
  }

  static fromString(id: string): TaskId {
    return new TaskId(id);
  }

  private validate(): void {
    if (!this.value || typeof this.value !== 'string' || this.value.trim().length === 0) {
      throw new Error('Task ID must be a non-empty string');
    }
  }
}