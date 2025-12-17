import { TaskId } from '../values/task-id.value';
import { TaskTitle } from '../values/task-title.value';
import { TaskStatusValue, TaskStatus } from '../values/task-status.enum';
import { TaskPriorityValue } from '../values/task-priority.enum';
import { TaskCreatedEvent } from '../events/task-created.event';
import { TaskUpdatedEvent } from '../events/task-updated.event';
import { TaskStatusChangedEvent } from '../events/task-status-changed.event';

export class Task {
  private readonly id: TaskId;
  private title: TaskTitle;
  private description: string | null;
  private status: TaskStatusValue;
  private priority: TaskPriorityValue;
  private readonly createdAt: Date;
  private updatedAt: Date;
  private uncommittedEvents: any[] = [];

  private constructor(
    id: TaskId,
    title: TaskTitle,
    status: TaskStatusValue,
    priority: TaskPriorityValue,
    description: string | null = null,
    createdAt: Date = new Date(),
    updatedAt: Date = new Date()
  ) {
    this.id = id;
    this.title = title;
    this.description = description;
    this.status = status;
    this.priority = priority;
    this.createdAt = createdAt;
    this.updatedAt = updatedAt;
  }

  static create(
    id: TaskId,
    title: TaskTitle,
    status: TaskStatusValue,
    priority: TaskPriorityValue,
    description?: string
  ): Task {
    const task = new Task(id, title, status, priority, description || null);
    task.addEvent(new TaskCreatedEvent(
      id.getValue(),
      title.getValue(),
      status.getValue(),
      priority.getValue(),
      description || null
    ));
    return task;
  }

  static createWithGeneratedId(
    title: TaskTitle,
    status: TaskStatusValue,
    priority: TaskPriorityValue,
    description?: string
  ): Task {
    const id = TaskId.generate();
    return Task.create(id, title, status, priority, description);
  }

  static reconstruct(
    id: TaskId,
    title: TaskTitle,
    status: TaskStatusValue,
    priority: TaskPriorityValue,
    description: string | null,
    createdAt: Date,
    updatedAt: Date
  ): Task {
    return new Task(id, title, status, priority, description, createdAt, updatedAt);
  }

  // Getters
  getId(): TaskId {
    return this.id;
  }

  getTitle(): TaskTitle {
    return this.title;
  }

  getDescription(): string | null {
    return this.description;
  }

  getStatus(): TaskStatusValue {
    return this.status;
  }

  getPriority(): TaskPriorityValue {
    return this.priority;
  }

  getCreatedAt(): Date {
    return new Date(this.createdAt);
  }

  getUpdatedAt(): Date {
    return new Date(this.updatedAt);
  }

  // Business methods
  updateTitle(newTitle: TaskTitle): void {
    if (!this.title.equals(newTitle)) {
      this.title = newTitle;
      this.updateTimestamp();
      this.addEvent(new TaskUpdatedEvent(
        this.id.getValue(),
        { title: newTitle.getValue() }
      ));
    }
  }

  updateDescription(newDescription: string | null): void {
    if (this.description !== newDescription) {
      this.description = newDescription;
      this.updateTimestamp();
      this.addEvent(new TaskUpdatedEvent(
        this.id.getValue(),
        { description: newDescription }
      ));
    }
  }

  updatePriority(newPriority: TaskPriorityValue): void {
    if (!this.priority.equals(newPriority)) {
      this.priority = newPriority;
      this.updateTimestamp();
      this.addEvent(new TaskUpdatedEvent(
        this.id.getValue(),
        { priority: newPriority.getValue() }
      ));
    }
  }

  changeStatus(newStatus: TaskStatusValue): void {
    if (this.status.equals(newStatus)) {
      return; // No-op if same status
    }

    if (!this.status.canTransitionTo(newStatus)) {
      throw new Error(
        `Cannot transition from ${this.status.getValue()} to ${newStatus.getValue()}`
      );
    }

    const oldStatus = this.status;
    this.status = newStatus;
    this.updateTimestamp();

    this.addEvent(new TaskStatusChangedEvent(
      this.id.getValue(),
      oldStatus.getValue(),
      newStatus.getValue()
    ));
  }

  // Status check methods
  isPending(): boolean {
    return this.status.getValue() === TaskStatus.PENDING;
  }

  isInProgress(): boolean {
    return this.status.getValue() === TaskStatus.IN_PROGRESS;
  }

  isCompleted(): boolean {
    return this.status.getValue() === TaskStatus.COMPLETED;
  }

  // Event handling
  getUncommittedEvents(): any[] {
    return [...this.uncommittedEvents];
  }

  clearUncommittedEvents(): void {
    this.uncommittedEvents = [];
  }

  // Equality
  equals(other: Task): boolean {
    if (other === null || other === undefined) {
      return false;
    }

    return this.id.equals(other.getId());
  }

  // Private helpers
  private addEvent(event: any): void {
    this.uncommittedEvents.push(event);
  }

  private updateTimestamp(): void {
    this.updatedAt = new Date();
  }
}