import type { ICommandHandler } from '@nestjs/cqrs';
import { CommandHandler, EventBus } from '@nestjs/cqrs';
import { CreateTaskCommand } from '../commands/create-task.command';
import type { TaskRepository } from '../../domain/repositories/task.repository.interface';
import { Task } from '../../domain/entities/task.entity';
import { TaskTitle } from '../../domain/values/task-title.value';
import { TaskStatusValue } from '../../domain/values/task-status.enum';

@CommandHandler(CreateTaskCommand)
export class CreateTaskHandler implements ICommandHandler<CreateTaskCommand> {
  constructor(
    private readonly taskRepository: TaskRepository,
    private readonly eventBus: EventBus
  ) {}

  async execute(command: CreateTaskCommand): Promise<void> {
    try {
      const task = Task.createWithGeneratedId(
        command.getTitle(),
        TaskStatusValue.pending(),
        command.getPriority(),
        command.getDescription() || undefined
      );

      await this.taskRepository.save(task);

      const events = task.getUncommittedEvents();
      if (events.length > 0) {
        this.eventBus.publishAll(events);
        task.clearUncommittedEvents();
      }
    } catch (error) {
      throw new Error(`Failed to create task: ${error.message}`);
    }
  }
}