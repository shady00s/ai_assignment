import { Injectable } from '@nestjs/common';
import { TaskRepository } from '../domain/repositories/task.repository.interface';
import { Task } from '../domain/entities/task.entity';
import { TaskId } from '../domain/values/task-id.value';
import { TaskTitle } from '../domain/values/task-title.value';
import { TaskStatusValue } from '../domain/values/task-status.enum';
import { TaskPriorityValue } from '../domain/values/task-priority.enum';
import { PrismaService } from '../../prisma.service';

@Injectable()
export class PrismaTaskRepository implements TaskRepository {
  constructor(private readonly prisma: PrismaService) {}

  async save(task: Task): Promise<void> {
    const data = {
      id: task.getId().getValue(),
      title: task.getTitle().getValue(),
      description: task.getDescription(),
      status: task.getStatus().getValue(),
      priority: task.getPriority().getValue(),
      createdAt: task.getCreatedAt(),
      updatedAt: task.getUpdatedAt(),
    };

    await this.prisma.task.upsert({
      where: { id: task.getId().getValue() },
      update: data,
      create: data,
    });
  }

  async findById(id: TaskId): Promise<Task | null> {
    const taskData = await this.prisma.task.findUnique({
      where: { id: id.getValue() },
    });

    if (!taskData) {
      return null;
    }

    return this.mapToDomainEntity(taskData);
  }

  async findAll(
    status?: TaskStatusValue,
    priority?: TaskPriorityValue
  ): Promise<Task[]> {
    const where: any = {};

    if (status) {
      where.status = status.getValue();
    }

    if (priority) {
      where.priority = priority.getValue();
    }

    const tasksData = await this.prisma.task.findMany({
      where,
      orderBy: {
        createdAt: 'desc',
      },
    });

    return tasksData.map(task => this.mapToDomainEntity(task));
  }

  async delete(id: TaskId): Promise<void> {
    await this.prisma.task.delete({
      where: { id: id.getValue() },
    });
  }

  async exists(id: TaskId): Promise<boolean> {
    const task = await this.prisma.task.findUnique({
      where: { id: id.getValue() },
      select: { id: true },
    });

    return !!task;
  }

  private mapToDomainEntity(taskData: any): Task {
    return Task.reconstruct(
      TaskId.fromString(taskData.id),
      TaskTitle.create(taskData.title),
      TaskStatusValue.fromString(taskData.status),
      TaskPriorityValue.fromString(taskData.priority),
      taskData.description,
      taskData.createdAt,
      taskData.updatedAt
    );
  }
}