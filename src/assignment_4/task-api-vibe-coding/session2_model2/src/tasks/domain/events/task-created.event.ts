export class TaskCreatedEvent {
  constructor(
    public readonly taskId: string,
    public readonly title: string,
    public readonly status: string,
    public readonly priority: string,
    public readonly description: string | null,
    public readonly occurredOn: Date = new Date()
  ) {}

  getEventName(): string {
    return 'TaskCreated';
  }
}