import { Injectable, NotFoundException, ForbiddenException, ConflictException } from '@nestjs/common';
import { DatabaseService } from '../config/database.config';
import { CreateTaskDto } from './dto/create-task.dto';
import { UpdateTaskDto } from './dto/update-task.dto';
import { Task, User } from '@prisma/client';
import { Task as FrontendTask } from './types';
import { UsersService } from '@/users/users.service';

// String constants for type safety - ALIGNED WITH FRONTEND
export const TaskStatus = {
  TODO: 'TODO',
  IN_PROGRESS: 'IN_PROGRESS',
  COMPLETED: 'COMPLETED',
  CANCELLED: 'CANCELLED',
} as const;

export const Priority = {
  LOW: 'LOW',
  MEDIUM: 'MEDIUM',
  HIGH: 'HIGH',
  CRITICAL: 'CRITICAL',
} as const;

@Injectable()
export class TasksService {
  constructor(private readonly prisma: DatabaseService,
    private readonly usersService: UsersService,
  ) {}

  private parseJsonField(field: string | null): any {
    if (!field) return null;
    try {
      return JSON.parse(field);
    } catch {
      return null;
    }
  }

  private parseJsonArrayField(field: string | null): any[] {
    if (!field) return [];
    try {
      return JSON.parse(field);
    } catch {
      return [];
    }
  }

  async create(createTaskDto: CreateTaskDto, userId: string): Promise<FrontendTask> {
    const createdTask = await this.prisma.task.create({
      data: {
        title: createTaskDto.title,
        description: createTaskDto.description,
        priority: createTaskDto.priority || Priority.MEDIUM,
        dueDate: createTaskDto.dueDate,
        estimatedPomodoros: createTaskDto.estimatedPomodoros || 1,
        complexity: createTaskDto.complexity || 1,
        tags: createTaskDto.tags ? JSON.stringify(createTaskDto.tags) : '[]',
        creatorId: userId,
        assigneeId: createTaskDto.assigneeId || userId,
        teamId: createTaskDto.teamId,
      },
      include: {
        creator: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        assignee: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        team: {
          select: { id: true, name: true },
        },
      },
    });

      // Return data in frontend-compatible format
    return {
      id: createdTask.id,
      title: createdTask.title,
      description: createdTask.description || undefined,
      status: createdTask.status as any, // Already aligned with frontend
      priority: createdTask.priority as any, // Already aligned with frontend
      estimatedPomodoros: createdTask.estimatedPomodoros,
      completedPomodoros: createdTask.completedPomodoros,
      assigneeId: createdTask.assigneeId || undefined,
      assignee: createdTask.assigneeId ? {
        id: createdTask.assigneeId,
        email: createdTask.assigneeId, // Would be populated with actual user data
        firstName: "User",
        lastName: "Name",
        avatar: undefined,
        teamId: undefined,
        level: 1,
        xp: 0,
        streak: 0,
        totalFocusTime: 0,
        tasksCompleted: 0,
        qualityScore: 0,
        wellnessScore: 0,
        preferences: {} as any,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      } : undefined,
      dueDate: createdTask.dueDate ? createdTask.dueDate.toISOString() : undefined,
      createdAt: createdTask.createdAt.toISOString(),
      updatedAt: createdTask.updatedAt.toISOString(),
      completedAt: createdTask.completedAt ? createdTask.completedAt.toISOString() : undefined,
      tags: this.parseJsonArrayField(createdTask.tags),
      projectId: createdTask.teamId || undefined, // Map teamId to projectId for frontend
    };
  }

  async findAll(
    userId: string,
    filters?: {
      status?: string | string[];
      priority?: string | string[];
      teamId?: string;
      assigneeId?: string;
      tags?: string | string[];
    },
    sort?: {
      field: 'createdAt' | 'updatedAt' | 'dueDate' | 'priority' | 'title' | 'status';
      direction: 'ASC' | 'DESC';
    }
  ): Promise<FrontendTask[]> {
    const whereClause: any = {
      OR: [
        { creatorId: userId },
        { assigneeId: userId },
      ],
    };

    if (filters?.status) {
      whereClause.status = Array.isArray(filters.status)
        ? { in: filters.status }
        : filters.status;
    }

    if (filters?.priority) {
      whereClause.priority = Array.isArray(filters.priority)
        ? { in: filters.priority }
        : filters.priority;
    }

    if (filters?.teamId) {
      whereClause.teamId = filters.teamId;
    }

    if (filters?.assigneeId) {
      whereClause.assigneeId = filters.assigneeId;
    }

    if (filters?.tags) {
      const tagArray = Array.isArray(filters.tags) ? filters.tags : [filters.tags];
      whereClause.tags = {
        contains: tagArray.map(tag => `"${tag}"`).join(',')
      };
    }

    const tasks = await this.prisma.task.findMany({
      where: whereClause,
      include: {
        creator: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        assignee: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        team: {
          select: { id: true, name: true },
        },
        dependencies: {
          include: {
            prerequisite: {
              select: { id: true, title: true },
            },
          },
        },
        dependents: {
          include: {
            dependentTask: {
              select: { id: true, title: true },
            },
          },
        },
      },
      orderBy: sort
        ? [{ [sort.field]: sort.direction.toLowerCase() }]
        : [
            { priority: 'desc' },
            { dueDate: 'asc' },
            { createdAt: 'desc' },
          ],
    });
 
      // Return data in frontend-compatible format
    return tasks.map(task => ({
      id: task.id,
      title: task.title,
      description: task.description || undefined,
      status: task.status as any,
      priority: task.priority as any,
      estimatedPomodoros: task.estimatedPomodoros,
      completedPomodoros: task.completedPomodoros,
      assigneeId: task.assigneeId || undefined,
      assignee: task.assignee ? {
        id: task.assignee.id,
        email: task.assignee.email,
        firstName: task.assignee.firstName,
        lastName: task.assignee.lastName,
        avatar: task.assignee.avatar || undefined,
        teamId: undefined,
        level: 1,
        xp: 0,
        streak: 0,
        totalFocusTime: 0,
        tasksCompleted: 0,
        qualityScore: 0,
        wellnessScore: 0,
        preferences: {} as any,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      } : undefined,
      dueDate: task.dueDate ? task.dueDate.toISOString() : undefined,
      createdAt: task.createdAt.toISOString(),
      updatedAt: task.updatedAt.toISOString(),
      completedAt: task.completedAt ? task.completedAt.toISOString() : undefined,
      tags: this.parseJsonArrayField(task.tags),
      projectId: task.teamId || undefined,
    }));
  }

  async findOne(id: string, userId: string): Promise<FrontendTask> {
    const task = await this.prisma.task.findUnique({
      where: { id },
      include: {
        creator: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        assignee: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        team: {
          select: { id: true, name: true },
        },
        dependencies: {
          include: {
            prerequisite: {
              select: { id: true, title: true },
            },
          },
        },
        dependents: {
          include: {
            dependentTask: {
              select: { id: true, title: true },
            },
          },
        },
        sessions: {
          select: { id: true, duration: true, startTime: true, endTime: true, quality: true },
          orderBy: { startTime: 'desc' },
        },
      },
    });

    if (!task) {
      throw new NotFoundException('Task not found');
    }

    // Check if user has access to this task
    const hasAccess =
      task.creatorId === userId ||
      task.assigneeId === userId ||
      (task.teamId && await this.isTeamMember(userId, task.teamId));

      const assignee = await this.usersService.findById(task.assigneeId);


    if (!hasAccess) {
      throw new ForbiddenException('Access denied to this task');
    }

    // Return data in frontend-compatible format
    return {
      id: task.id,
      title: task.title,
      description: task.description || undefined,
      status: task.status as any,
      priority: task.priority as any,
      estimatedPomodoros: task.estimatedPomodoros,
      completedPomodoros: task.completedPomodoros,
      assigneeId: task.assigneeId || undefined,
      dueDate: task.dueDate ? task.dueDate.toISOString() : undefined,
      createdAt: task.createdAt.toISOString(),
      updatedAt: task.updatedAt.toISOString(),
      completedAt: task.completedAt ? task.completedAt.toISOString() : undefined,
      tags: this.parseJsonArrayField(task.tags),
      projectId: task.teamId || undefined,
    };
  }

  async update(id: string, updateTaskDto: UpdateTaskDto, userId: string): Promise<FrontendTask> {
    const task = await this.findOne(id, userId);

    // Handle status changes and automatic field updates
    const updateData: any = { ...updateTaskDto };

    // Handle tags field - serialize to JSON string for SQLite
    if (updateTaskDto.tags !== undefined) {
      updateData.tags = updateTaskDto.tags ? JSON.stringify(updateTaskDto.tags) : '[]';
    }

    // If task is being marked as completed, set completion timestamp
    if (updateTaskDto.status === TaskStatus.COMPLETED && task.status !== TaskStatus.COMPLETED) {
      updateData.completedAt = new Date();
      // Calculate actual minutes from sessions
      const sessionResult = await this.prisma.session.aggregate({
        where: {
          taskId: id,
          completed: true,
        },
        _sum: { duration: true },
      });
      updateData.actualMinutes = sessionResult._sum.duration || 0;
    }

    // Update completed pomodoros based on sessions
    if (updateTaskDto.status !== undefined) {
      const completedSessions = await this.prisma.session.count({
        where: {
          taskId: id,
          completed: true,
        },
      });
      updateData.completedPomodoros = completedSessions;
    }

    const updatedTask = await this.prisma.task.update({
      where: { id },
      data: updateData,
      include: {
        creator: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        assignee: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        team: {
          select: { id: true, name: true },
        },
      },
    });

      // Return data in frontend-compatible format
    return {
      id: updatedTask.id,
      title: updatedTask.title,
      description: updatedTask.description || undefined,
      status: updatedTask.status as any,
      priority: updatedTask.priority as any,
      estimatedPomodoros: updatedTask.estimatedPomodoros,
      completedPomodoros: updatedTask.completedPomodoros,
      assigneeId: updatedTask.assigneeId || undefined,
      assignee: updatedTask.assignee ? {
        id: updatedTask.assignee.id,
        email: updatedTask.assignee.email,
        firstName: updatedTask.assignee.firstName,
        lastName: updatedTask.assignee.lastName,
        avatar: updatedTask.assignee.avatar || undefined,
        teamId: undefined,
        level: 1,
        xp: 0,
        streak: 0,
        totalFocusTime: 0,
        tasksCompleted: 0,
        qualityScore: 0,
        wellnessScore: 0,
        preferences: {} as any,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      } : undefined,
      dueDate: updatedTask.dueDate ? updatedTask.dueDate.toISOString() : undefined,
      createdAt: updatedTask.createdAt.toISOString(),
      updatedAt: updatedTask.updatedAt.toISOString(),
      completedAt: updatedTask.completedAt ? updatedTask.completedAt.toISOString() : undefined,
      tags: this.parseJsonArrayField(updatedTask.tags),
      projectId: updatedTask.teamId || undefined,
    };
  }

  async remove(id: string, userId: string): Promise<FrontendTask> {
    const task = await this.findOne(id, userId);

    // Only creator can delete the task
    if (task.assigneeId !== userId) {
      throw new ForbiddenException('Only the task assignee can delete this task');
    }

    const deletedTask = await this.prisma.task.delete({
      where: { id },
    });

    // Return the deleted task in frontend format for confirmation
    return {
      id: deletedTask.id,
      title: deletedTask.title,
      description: deletedTask.description || undefined,
      status: deletedTask.status as any,
      priority: deletedTask.priority as any,
      estimatedPomodoros: deletedTask.estimatedPomodoros,
      completedPomodoros: deletedTask.completedPomodoros,
      assigneeId: deletedTask.assigneeId || undefined,
      assignee: undefined, // Deleted task won't have relations
      dueDate: deletedTask.dueDate ? deletedTask.dueDate.toISOString() : undefined,
      createdAt: deletedTask.createdAt.toISOString(),
      updatedAt: deletedTask.updatedAt.toISOString(),
      completedAt: deletedTask.completedAt ? deletedTask.completedAt.toISOString() : undefined,
      tags: this.parseJsonArrayField(deletedTask.tags),
      projectId: deletedTask.teamId || undefined,
    };
  }

  async addDependency(taskId: string, prerequisiteId: string, userId: string) {
    // Verify both tasks exist and user has access
    const [task, prerequisite] = await Promise.all([
      this.findOne(taskId, userId),
      this.findOne(prerequisiteId, userId),
    ]);

    // Prevent circular dependencies
    if (await this.wouldCreateCircularDependency(taskId, prerequisiteId)) {
      throw new ConflictException('This would create a circular dependency');
    }

    return this.prisma.taskDependency.create({
      data: {
        dependentTaskId: taskId,
        prerequisiteId,
      },
      include: {
        dependentTask: {
          select: { id: true, title: true },
        },
        prerequisite: {
          select: { id: true, title: true },
        },
      },
    });
  }

  async removeDependency(taskId: string, prerequisiteId: string, userId: string) {
    const task = await this.findOne(taskId, userId);

    const dependency = await this.prisma.taskDependency.findUnique({
      where: {
        dependentTaskId_prerequisiteId: {
          dependentTaskId: taskId,
          prerequisiteId,
        },
      },
    });

    if (!dependency) {
      throw new NotFoundException('Dependency not found');
    }

    return this.prisma.taskDependency.delete({
      where: {
        dependentTaskId_prerequisiteId: {
          dependentTaskId: taskId,
          prerequisiteId,
        },
      },
    });
  }

  async getTasksByProject(userId: string, projectId: string): Promise<FrontendTask[]> {
    // This would integrate with a projects table when implemented
    return this.findAll(userId, { teamId: projectId });
  }

  async getTaskAnalytics(userId: string, startDate?: Date, endDate?: Date) {
    const whereClause: any = {
      OR: [
        { creatorId: userId },
        { assigneeId: userId },
      ],
    };

    if (startDate || endDate) {
      whereClause.createdAt = {
        gte: startDate,
        lte: endDate,
      };
    }

    const [
      totalTasks,
      completedTasks,
      tasksByStatus,
      tasksByPriority,
      overdueTasks,
      upcomingTasks,
    ] = await Promise.all([
      this.prisma.task.count({ where: whereClause }),
      this.prisma.task.count({
        where: { ...whereClause, status: TaskStatus.COMPLETED },
      }),
      this.prisma.task.groupBy({
        by: ['status'],
        where: whereClause,
        _count: true,
      }),
      this.prisma.task.groupBy({
        by: ['priority'],
        where: whereClause,
        _count: true,
      }),
      this.prisma.task.count({
        where: {
          ...whereClause,
          dueDate: { lt: new Date() },
          status: { not: TaskStatus.COMPLETED },
        },
      }),
      this.prisma.task.findMany({
        where: {
          ...whereClause,
          dueDate: {
            gte: new Date(),
            lte: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // Next 7 days
          },
          status: { not: TaskStatus.COMPLETED },
        },
        orderBy: { dueDate: 'asc' },
        take: 10,
      }),
    ]);

    const completionRate = totalTasks > 0 ? (completedTasks / totalTasks) * 100 : 0;

    return {
      totalTasks,
      completedTasks,
      completionRate,
      tasksByStatus: tasksByStatus.reduce((acc, item) => {
        acc[item.status] = item._count;
        return acc;
      }, {} as Record<string, number>),
      tasksByPriority: tasksByPriority.reduce((acc, item) => {
        acc[item.priority] = item._count;
        return acc;
      }, {} as Record<string, number>),
      overdueTasks,
      upcomingDeadlines: upcomingTasks,
    };
  }

  private async isTeamMember(userId: string, teamId: string): Promise<boolean> {
    const membership = await this.prisma.teamMember.findUnique({
      where: {
        userId_teamId: {
          userId,
          teamId,
        },
      },
    });

    return !!membership;
  }

  private async wouldCreateCircularDependency(
    taskId: string,
    prerequisiteId: string,
    visited = new Set<string>()
  ): Promise<boolean> {
    if (visited.has(prerequisiteId)) {
      return true;
    }

    visited.add(prerequisiteId);

    const dependencies = await this.prisma.taskDependency.findMany({
      where: { dependentTaskId: prerequisiteId },
      include: { prerequisite: true },
    });

    for (const dep of dependencies) {
      if (dep.prerequisiteId === taskId ||
          await this.wouldCreateCircularDependency(taskId, dep.prerequisiteId, visited)) {
        return true;
      }
    }

    return false;
  }
}