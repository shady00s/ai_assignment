import { QueryHandler, IQueryHandler } from '@nestjs/cqrs';
import { GetTasksQuery } from '../queries/get-tasks.query';
import { TaskRepository } from '../../domain/repositories/task.repository.interface';
import { TaskResponseDto } from '../dtos/task-response.dto';

@QueryHandler(GetTasksQuery)
export class GetTasksHandler implements IQueryHandler<GetTasksQuery, TaskResponseDto[]> {
  constructor(private readonly taskRepository: TaskRepository) {}

  async execute(query: GetTasksQuery): Promise<TaskResponseDto[]> {
    try {
      const tasks = await this.taskRepository.findAll(
        query.getStatus(),
        query.getPriority()
      );

      return tasks.map(task => TaskResponseDto.fromDomainEntity(task));
    } catch (error) {
      throw new Error(`Failed to retrieve tasks: ${error.message}`);
    }
  }
}