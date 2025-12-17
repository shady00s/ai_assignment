export class TaskStatusChangedEvent {
  constructor(
    public readonly taskId: string,
    public readonly previousStatus: string,
    public readonly newStatus: string,
    public readonly occurredOn: Date = new Date()
  ) {}

  getEventName(): string {
    return 'TaskStatusChanged';
  }
}