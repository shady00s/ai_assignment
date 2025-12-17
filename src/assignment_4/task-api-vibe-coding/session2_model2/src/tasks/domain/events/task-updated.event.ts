export class TaskUpdatedEvent {
  constructor(
    public readonly taskId: string,
    public readonly updatedFields: Record<string, any>,
    public readonly occurredOn: Date = new Date()
  ) {}

  getEventName(): string {
    return 'TaskUpdated';
  }
}