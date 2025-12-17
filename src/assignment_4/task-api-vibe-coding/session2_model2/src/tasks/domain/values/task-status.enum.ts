export enum TaskStatus {
  PENDING = 'PENDING',
  IN_PROGRESS = 'IN_PROGRESS',
  COMPLETED = 'COMPLETED'
}

export class TaskStatusValue {
  private readonly value: TaskStatus;

  private constructor(value: TaskStatus) {
    this.value = value;
  }

  static pending(): TaskStatusValue {
    return new TaskStatusValue(TaskStatus.PENDING);
  }

  static inProgress(): TaskStatusValue {
    return new TaskStatusValue(TaskStatus.IN_PROGRESS);
  }

  static completed(): TaskStatusValue {
    return new TaskStatusValue(TaskStatus.COMPLETED);
  }

  static fromString(status: string): TaskStatusValue {
    const upperStatus = status.toUpperCase();

    switch (upperStatus) {
      case TaskStatus.PENDING:
        return TaskStatusValue.pending();
      case TaskStatus.IN_PROGRESS:
        return TaskStatusValue.inProgress();
      case TaskStatus.COMPLETED:
        return TaskStatusValue.completed();
      default:
        throw new Error(`Invalid task status: ${status}`);
    }
  }

  getValue(): TaskStatus {
    return this.value;
  }

  equals(other: TaskStatusValue): boolean {
    return this.value === other.value;
  }

  toString(): string {
    return this.value;
  }

  canTransitionTo(newStatus: TaskStatusValue): boolean {
    const transitions = {
      [TaskStatus.PENDING]: [TaskStatus.PENDING, TaskStatus.IN_PROGRESS],
      [TaskStatus.IN_PROGRESS]: [TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED],
      [TaskStatus.COMPLETED]: [TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS]
    };

    return transitions[this.value].includes(newStatus.value);
  }
}