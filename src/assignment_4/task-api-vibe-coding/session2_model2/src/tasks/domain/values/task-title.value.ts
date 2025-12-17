import { ValueObject } from './base.value-object';

export class TaskTitle extends ValueObject<string> {
  private constructor(value: string) {
    super(value.trim());
    this.validate();
  }

  static create(title: string): TaskTitle {
    return new TaskTitle(title);
  }

  private validate(): void {
    if (!this.value || this.value.length === 0) {
      throw new Error('Task title cannot be empty');
    }

    if (this.value.length > 255) {
      throw new Error('Task title cannot exceed 255 characters');
    }

    if (this.value.trim().length === 0) {
      throw new Error('Task title cannot be whitespace only');
    }
  }
}