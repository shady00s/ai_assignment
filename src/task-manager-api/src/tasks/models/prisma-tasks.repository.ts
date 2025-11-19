import { Injectable, NotFoundException, ConflictException } from '@nestjs/common';
import { PrismaService } from '../../prisma/prisma.service';
import { Task, TaskStatus } from '../entities/task.entity';
import { CreateTaskDto } from '../dto/create-task.dto';
import { UpdateTaskDto } from '../dto/update-task.dto';
import { QueryTasksDto } from '../dto/query-tasks.dto';

@Injectable()
export class PrismaTasksRepository {
  constructor(private readonly prisma: PrismaService) {}

  async findAll(query: QueryTasksDto): Promise<Task[]> {
    const where: any = {};

    if (query.status) {
      where.status = query.status;
    }

    if (query.search) {
      where.OR = [
        { title: { contains: query.search } },
        { description: { contains: query.search } }
      ];
    }

    const orderBy: any = {};
    const sortBy = query.sortBy || 'createdAt';
    const sortOrder = query.sortOrder || 'desc';
    orderBy[sortBy] = sortOrder;

    const take = query.limit || 10;
    const skip = ((query.page || 1) - 1) * take;

    const tasks = await this.prisma.task.findMany({
      where,
      orderBy,
      take,
      skip,
    });

    return tasks.map(task => ({
      id: task.id,
      title: task.title,
      description: task.description,
      status: task.status as TaskStatus,
      created_at: task.createdAt,
      updated_at: task.updatedAt,
    }));
  }

  async findById(id: string): Promise<Task | null> {
    const task = await this.prisma.task.findUnique({
      where: { id },
    });

    if (!task) {
      return null;
    }

    return {
      id: task.id,
      title: task.title,
      description: task.description,
      status: task.status as TaskStatus,
      created_at: task.createdAt,
      updated_at: task.updatedAt,
    };
  }

  async create(createTaskDto: CreateTaskDto): Promise<Task> {
    try {
      const task = await this.prisma.task.create({
        data: {
          title: createTaskDto.title,
          description: createTaskDto.description,
          status: createTaskDto.status || 'PENDING',
        },
      });

      return {
        id: task.id,
        title: task.title,
        description: task.description,
        status: task.status as TaskStatus,
        created_at: task.createdAt,
        updated_at: task.updatedAt,
      };
    } catch (error) {
      if (error.code === 'P2002') {
        throw new ConflictException('Task already exists');
      }
      throw error;
    }
  }

  async update(id: string, updateTaskDto: UpdateTaskDto): Promise<Task | null> {
    try {
      const task = await this.prisma.task.update({
        where: { id },
        data: {
          ...(updateTaskDto.title && { title: updateTaskDto.title }),
          ...(updateTaskDto.description !== undefined && { description: updateTaskDto.description }),
          ...(updateTaskDto.status && { status: updateTaskDto.status }),
        },
      });

      return {
        id: task.id,
        title: task.title,
        description: task.description,
        status: task.status as TaskStatus,
        created_at: task.createdAt,
        updated_at: task.updatedAt,
      };
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Task with ID "${id}" not found`);
      }
      if (error.code === 'P2002') {
        throw new ConflictException('Task already exists');
      }
      throw error;
    }
  }

  async delete(id: string): Promise<boolean> {
    try {
      await this.prisma.task.delete({
        where: { id },
      });
      return true;
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Task with ID "${id}" not found`);
      }
      throw error;
    }
  }

  async findByStatus(status: TaskStatus): Promise<Task[]> {
    const tasks = await this.prisma.task.findMany({
      where: { status },
      orderBy: { createdAt: 'desc' },
    });

    return tasks.map(task => ({
      id: task.id,
      title: task.title,
      description: task.description,
      status: task.status as TaskStatus,
      created_at: task.createdAt,
      updated_at: task.updatedAt,
    }));
  }

  async searchTasks(query: string): Promise<Task[]> {
    const tasks = await this.prisma.task.findMany({
      where: {
        OR: [
          { title: { contains: query } },
          { description: { contains: query } }
        ],
      },
      orderBy: { createdAt: 'desc' },
    });

    return tasks.map(task => ({
      id: task.id,
      title: task.title,
      description: task.description,
      status: task.status as TaskStatus,
      created_at: task.createdAt,
      updated_at: task.updatedAt,
    }));
  }

  async getStats(): Promise<{ total: number; pending: number; completed: number }> {
    const [total, pending, completed] = await Promise.all([
      this.prisma.task.count(),
      this.prisma.task.count({ where: { status: 'PENDING' } }),
      this.prisma.task.count({ where: { status: 'COMPLETED' } }),
    ]);

    return { total, pending, completed };
  }

  async bulkCreate(createTaskDtos: CreateTaskDto[]): Promise<Task[]> {
    const tasks = await this.prisma.$transaction(
      createTaskDtos.map(dto =>
        this.prisma.task.create({
          data: {
            title: dto.title,
            description: dto.description,
            status: dto.status || 'PENDING',
          },
        })
      )
    );

    return tasks.map(task => ({
      id: task.id,
      title: task.title,
      description: task.description,
      status: task.status as TaskStatus,
      created_at: task.createdAt,
      updated_at: task.updatedAt,
    }));
  }

  async bulkUpdateStatus(ids: string[], status: TaskStatus): Promise<{ count: number }> {
    const result = await this.prisma.task.updateMany({
      where: { id: { in: ids } },
      data: { status },
    });

    return { count: result.count };
  }
}