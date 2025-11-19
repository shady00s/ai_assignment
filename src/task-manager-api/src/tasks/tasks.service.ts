import { Injectable, NotFoundException, BadRequestException, Inject } from '@nestjs/common';
import { Task, TaskStatus } from './entities/task.entity';
import { CreateTaskDto } from './dto/create-task.dto';
import { UpdateTaskDto } from './dto/update-task.dto';
import { QueryTasksDto } from './dto/query-tasks.dto';
import { TasksRepository } from './models/tasks.repository';
import { PrismaTasksRepository } from './models/prisma-tasks.repository';

@Injectable()
export class TasksService {
  constructor(
    @Inject('USE_JSON_STORAGE') private readonly useJsonStorage: boolean,
    private readonly jsonRepository: TasksRepository,
    private readonly prismaRepository: PrismaTasksRepository,
  ) {}

  private getRepository() {
    return this.useJsonStorage ? this.jsonRepository : this.prismaRepository;
  }

  async create(createTaskDto: CreateTaskDto): Promise<Task> {
    if (this.useJsonStorage) {
      const task: Task = {
        id: '',
        title: createTaskDto.title,
        description: createTaskDto.description,
        status: createTaskDto.status || TaskStatus.PENDING,
        created_at: new Date(),
        updated_at: new Date(),
      };

      return await this.jsonRepository.save(task);
    } else {
      return await this.prismaRepository.create(createTaskDto);
    }
  }

  async findAll(query: QueryTasksDto): Promise<Task[]> {
    return await this.getRepository().findAll(query);
  }

  async findOne(id: string): Promise<Task> {
    const task = await this.getRepository().findById(id);
    if (!task) {
      throw new NotFoundException(`Task with ID "${id}" not found`);
    }
    return task;
  }

  async update(id: string, updateTaskDto: UpdateTaskDto): Promise<Task> {
    const existingTask = await this.getRepository().findById(id);
    if (!existingTask) {
      throw new NotFoundException(`Task with ID "${id}" not found`);
    }

    if (Object.keys(updateTaskDto).length === 0) {
      throw new BadRequestException('No valid fields provided for update');
    }

    if (this.useJsonStorage) {
      return await this.jsonRepository.update(id, updateTaskDto);
    } else {
      const updatedTask = await this.prismaRepository.update(id, updateTaskDto);
      if (!updatedTask) {
        throw new NotFoundException(`Task with ID "${id}" not found`);
      }
      return updatedTask;
    }
  }

  async remove(id: string): Promise<void> {
    const task = await this.getRepository().findById(id);
    if (!task) {
      throw new NotFoundException(`Task with ID "${id}" not found`);
    }

    if (this.useJsonStorage) {
      const deleted = await this.jsonRepository.delete(id);
      if (!deleted) {
        throw new BadRequestException(`Failed to delete task with ID "${id}"`);
      }
    } else {
      await this.prismaRepository.delete(id);
    }
  }

  async getStats(): Promise<{
    total: number;
    pending: number;
    completed: number;
  }> {
    if (this.useJsonStorage) {
      const tasks = await this.jsonRepository.findAll();
      return {
        total: tasks.length,
        pending: tasks.filter(task => task.status === TaskStatus.PENDING).length,
        completed: tasks.filter(task => task.status === TaskStatus.COMPLETED).length,
      };
    } else {
      return await this.prismaRepository.getStats();
    }
  }

  // Advanced methods for Prisma backend
  async bulkCreate(createTaskDtos: CreateTaskDto[]): Promise<Task[]> {
    if (this.useJsonStorage) {
      throw new BadRequestException('Bulk operations are only supported with database storage');
    }
    return await this.prismaRepository.bulkCreate(createTaskDtos);
  }

  async bulkUpdateStatus(ids: string[], status: TaskStatus): Promise<{ count: number }> {
    if (this.useJsonStorage) {
      throw new BadRequestException('Bulk operations are only supported with database storage');
    }
    return await this.prismaRepository.bulkUpdateStatus(ids, status);
  }

  // Method to check current storage backend
  getStorageBackend(): 'json' | 'database' {
    return this.useJsonStorage ? 'json' : 'database';
  }
}