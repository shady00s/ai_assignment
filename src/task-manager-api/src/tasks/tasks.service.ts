import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { Task, TaskStatus } from './entities/task.entity';
import { CreateTaskDto } from './dto/create-task.dto';
import { UpdateTaskDto } from './dto/update-task.dto';
import { QueryTasksDto } from './dto/query-tasks.dto';
import { TasksRepository } from './models/tasks.repository';

@Injectable()
export class TasksService {
  constructor(private readonly tasksRepository: TasksRepository) {}

  async create(createTaskDto: CreateTaskDto): Promise<Task> {
    const task: Task = {
      id: '',
      title: createTaskDto.title,
      description: createTaskDto.description,
      status: createTaskDto.status || TaskStatus.PENDING,
      created_at: new Date(),
      updated_at: new Date(),
    };

    return await this.tasksRepository.save(task);
  }

  async findAll(query: QueryTasksDto): Promise<Task[]> {
    let tasks = await this.tasksRepository.findAll();

    if (query.status) {
      tasks = tasks.filter(task => task.status === query.status);
    }

    if (query.search) {
      const searchQuery = query.search.toLowerCase();
      tasks = tasks.filter(task =>
        task.title.toLowerCase().includes(searchQuery) ||
        (task.description && task.description.toLowerCase().includes(searchQuery))
      );
    }

    const sortBy = query.sortBy || 'created_at';
    const sortOrder = query.sortOrder || 'desc';

    tasks.sort((a, b) => {
      const aValue = a[sortBy];
      const bValue = b[sortBy];

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    const page = query.page || 1;
    const limit = query.limit || 10;
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;

    return tasks.slice(startIndex, endIndex);
  }

  async findOne(id: string): Promise<Task> {
    const task = await this.tasksRepository.findById(id);
    if (!task) {
      throw new NotFoundException(`Task with ID "${id}" not found`);
    }
    return task;
  }

  async update(id: string, updateTaskDto: UpdateTaskDto): Promise<Task> {
    const existingTask = await this.tasksRepository.findById(id);
    if (!existingTask) {
      throw new NotFoundException(`Task with ID "${id}" not found`);
    }

    if (Object.keys(updateTaskDto).length === 0) {
      throw new BadRequestException('No valid fields provided for update');
    }

    return await this.tasksRepository.update(id, updateTaskDto);
  }

  async remove(id: string): Promise<void> {
    const task = await this.tasksRepository.findById(id);
    if (!task) {
      throw new NotFoundException(`Task with ID "${id}" not found`);
    }

    const deleted = await this.tasksRepository.delete(id);
    if (!deleted) {
      throw new BadRequestException(`Failed to delete task with ID "${id}"`);
    }
  }

  async getStats(): Promise<{
    total: number;
    pending: number;
    completed: number;
  }> {
    const tasks = await this.tasksRepository.findAll();
    return {
      total: tasks.length,
      pending: tasks.filter(task => task.status === TaskStatus.PENDING).length,
      completed: tasks.filter(task => task.status === TaskStatus.COMPLETED).length,
    };
  }
}