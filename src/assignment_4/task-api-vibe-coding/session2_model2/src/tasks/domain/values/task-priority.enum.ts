export enum TaskPriority {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH'
}

export class TaskPriorityValue {
  private readonly value: TaskPriority;

  private constructor(value: TaskPriority) {
    this.value = value;
  }

  static low(): TaskPriorityValue {
    return new TaskPriorityValue(TaskPriority.LOW);
  }

  static medium(): TaskPriorityValue {
    return new TaskPriorityValue(TaskPriority.MEDIUM);
  }

  static high(): TaskPriorityValue {
    return new TaskPriorityValue(TaskPriority.HIGH);
  }

  static fromString(priority: string): TaskPriorityValue {
    const upperPriority = priority.toUpperCase();

    switch (upperPriority) {
      case TaskPriority.LOW:
        return TaskPriorityValue.low();
      case TaskPriority.MEDIUM:
        return TaskPriorityValue.medium();
      case TaskPriority.HIGH:
        return TaskPriorityValue.high();
      default:
        throw new Error(`Invalid task priority: ${priority}`);
    }
  }

  static default(): TaskPriorityValue {
    return TaskPriorityValue.medium();
  }

  getValue(): TaskPriority {
    return this.value;
  }

  equals(other: TaskPriorityValue): boolean {
    return this.value === other.value;
  }

  toString(): string {
    return this.value;
  }

  getNumericValue(): number {
    switch (this.value) {
      case TaskPriority.LOW:
        return 1;
      case TaskPriority.MEDIUM:
        return 2;
      case TaskPriority.HIGH:
        return 3;
      default:
        return 0;
    }
  }

  isHigherThan(other: TaskPriorityValue): boolean {
    return this.getNumericValue() > other.getNumericValue();
  }

  isLowerThan(other: TaskPriorityValue): boolean {
    return this.getNumericValue() < other.getNumericValue();
  }
}