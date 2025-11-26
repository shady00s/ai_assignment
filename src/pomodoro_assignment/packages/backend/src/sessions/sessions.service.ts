import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { DatabaseService } from '../config/database.config';
import { CreateSessionDto } from './dto/create-session.dto';
import { UpdateSessionDto } from './dto/update-session.dto';
import { Session } from '@prisma/client';

// String constants for type safety
export const SessionType = {
  POMODORO: 'POMODORO',
  SHORT_BREAK: 'SHORT_BREAK',
  LONG_BREAK: 'LONG_BREAK',
  CUSTOM: 'CUSTOM',
} as const;

export type SessionType = typeof SessionType[keyof typeof SessionType];

@Injectable()
export class SessionsService {
  constructor(private readonly prisma: DatabaseService) {}

  async createSession(createSessionDto: CreateSessionDto, userId: string): Promise<Session> {
    return this.prisma.session.create({
      data: {
        userId,
        taskId: createSessionDto.taskId,
        type: createSessionDto.type || 'POMODORO',
        duration: createSessionDto.duration,
        // startTime will be set when session is actually started
        notes: createSessionDto.notes,
      },
      include: {
        user: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        task: {
          select: { id: true, title: true },
        },
      },
    });
  }

  async findAll(
    userId: string,
    filters?: {
      type?: SessionType;
      taskId?: string;
      startDate?: Date;
      endDate?: Date;
    }
  ): Promise<Session[]> {
    const whereClause: any = { userId };

    if (filters?.type) {
      whereClause.type = filters.type;
    }

    if (filters?.taskId) {
      whereClause.taskId = filters.taskId;
    }

    if (filters?.startDate || filters?.endDate) {
      whereClause.startTime = {
        gte: filters.startDate,
        lte: filters.endDate,
      };
    }

    return this.prisma.session.findMany({
      where: whereClause,
      include: {
        user: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        task: {
          select: { id: true, title: true },
        },
      },
      orderBy: { startTime: 'desc' },
    });
  }

  async findOne(id: string, userId: string): Promise<Session> {
    const session = await this.prisma.session.findUnique({
      where: { id },
      include: {
        user: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        task: {
          select: { id: true, title: true },
        },
      },
    });

    if (!session) {
      throw new NotFoundException('Session not found');
    }

    if (session.userId !== userId) {
      throw new ForbiddenException('Access denied to this session');
    }

    return session;
  }

  async completeSession(id: string, userId: string, quality?: number, notes?: string): Promise<Session> {
    const session = await this.findOne(id, userId);

    if (session.completed) {
      throw new ForbiddenException('Session already completed');
    }

    const updatedSession = await this.prisma.session.update({
      where: { id },
      data: {
        endTime: new Date(),
        completed: true,
        quality: quality || null,
        notes: notes || undefined,
      },
      include: {
        user: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        task: {
          select: { id: true, title: true },
        },
      },
    });

    // Update task completion if this session completes the task
    if (updatedSession.taskId) {
      await this.updateTaskProgress(updatedSession.taskId);
    }

    return updatedSession;
  }

  async update(id: string, updateSessionDto: UpdateSessionDto, userId: string): Promise<Session> {
    await this.findOne(id, userId);

    return this.prisma.session.update({
      where: { id },
      data: updateSessionDto,
      include: {
        user: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        task: {
          select: { id: true, title: true },
        },
      },
    });
  }

  async remove(id: string, userId: string): Promise<Session> {
    const session = await this.findOne(id, userId);

    return this.prisma.session.delete({
      where: { id },
    });
  }

  async getSessionAnalytics(userId: string, startDate?: Date, endDate?: Date) {
    const whereClause: any = { userId };

    if (startDate || endDate) {
      whereClause.startTime = {
        gte: startDate,
        lte: endDate,
      };
    }

    const [
      totalSessions,
      completedSessions,
      totalMinutes,
      averageQuality,
      sessionsByType,
      recentSessions,
      currentStreak,
    ] = await Promise.all([
      this.prisma.session.count({ where: whereClause }),
      this.prisma.session.count({
        where: { ...whereClause, completed: true },
      }),
      this.prisma.session.aggregate({
        where: { ...whereClause, completed: true },
        _sum: { duration: true },
      }),
      this.prisma.session.aggregate({
        where: { ...whereClause, completed: true, quality: { not: null } },
        _avg: { quality: true },
      }),
      this.prisma.session.groupBy({
        by: ['type'],
        where: whereClause,
        _count: true,
      }),
      this.prisma.session.findMany({
        where: whereClause,
        orderBy: { startTime: 'desc' },
        take: 10,
        include: {
          task: { select: { id: true, title: true } },
        },
      }),
      this.calculateCurrentStreak(userId),
    ]);

    const completionRate = totalSessions > 0 ? (completedSessions / totalSessions) * 100 : 0;
    const totalMinutesCompleted = totalMinutes._sum.duration || 0;
    const avgQuality = averageQuality._avg.quality || 0;

    return {
      totalSessions,
      completedSessions,
      completionRate,
      totalMinutes: totalMinutesCompleted,
      averageQuality: avgQuality,
      sessionsByType: sessionsByType.reduce((acc, item) => {
        acc[item.type] = item._count;
        return acc;
      }, {} as Record<string, number>),
      recentSessions,
      currentStreak,
    };
  }

  async getActiveSession(userId: string): Promise<Session | null> {
    return this.prisma.session.findFirst({
      where: {
        userId,
        completed: false,
        endTime: null,
        startTime: { not: null }, // Only consider sessions that have actually started
      },
      include: {
        user: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        task: {
          select: { id: true, title: true },
        },
      },
      orderBy: { startTime: 'desc' },
    });
  }

  private async updateTaskProgress(taskId: string) {
    // Count completed sessions for this task
    const completedSessions = await this.prisma.session.count({
      where: {
        taskId,
        completed: true,
      },
    });

    // Get task details to check estimated pomodoros
    const task = await this.prisma.task.findUnique({
      where: { id: taskId },
      select: { estimatedPomodoros: true },
    });

    if (task && completedSessions >= task.estimatedPomodoros) {
      // Auto-mark task as complete if all pomodoros are done
      await this.prisma.task.update({
        where: { id: taskId },
        data: {
          status: 'DONE',
          completedAt: new Date(),
          completedPomodoros: completedSessions,
        },
      });
    } else {
      // Update completed pomodoros count
      await this.prisma.task.update({
        where: { id: taskId },
        data: {
          completedPomodoros: completedSessions,
        },
      });
    }
  }

  private async calculateCurrentStreak(userId: string): Promise<number> {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Check if there's a session completed today
    const todaySession = await this.prisma.session.findFirst({
      where: {
        userId,
        completed: true,
        endTime: {
          gte: today,
        },
      },
    });

    if (!todaySession) {
      // No session today, check yesterday
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);

      const yesterdaySession = await this.prisma.session.findFirst({
        where: {
          userId,
          completed: true,
          endTime: {
            gte: yesterday,
            lt: today,
          },
        },
      });

      return yesterdaySession ? 0 : 0; // Streak broken
    }

    // Count consecutive days with completed sessions
    let streak = 1;
    let currentDate = today;

    while (true) {
      currentDate.setDate(currentDate.getDate() - 1);
      const prevDate = new Date(currentDate);

      const prevSession = await this.prisma.session.findFirst({
        where: {
          userId,
          completed: true,
          endTime: {
            gte: prevDate,
            lt: new Date(prevDate.getTime() + 24 * 60 * 60 * 1000),
          },
        },
      });

      if (!prevSession) {
        break;
      }

      streak++;
    }

    return streak;
  }

  async startSession(id: string, userId: string): Promise<Session> {
    const session = await this.findOne(id, userId);

    if (session.startTime && !session.endTime) {
      throw new ForbiddenException('Session already started');
    }

    const updatedSession = await this.prisma.session.update({
      where: { id },
      data: {
        startTime: new Date(),
      },
      include: {
        user: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        task: {
          select: { id: true, title: true },
        },
      },
    });

    return updatedSession;
  }

  async pauseSession(id: string, userId: string): Promise<Session> {
    const session = await this.findOne(id, userId);

    if (!session.startTime || session.endTime) {
      throw new ForbiddenException('Session cannot be paused (not started or already completed)');
    }

    const updatedSession = await this.prisma.session.update({
      where: { id },
      data: {
        endTime: new Date(),
        completed: false, // Mark as paused, not completed
      },
      include: {
        user: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        task: {
          select: { id: true, title: true },
        },
      },
    });

    return updatedSession;
  }

  async skipSession(id: string, userId: string, notes?: string): Promise<Session> {
    const session = await this.findOne(id, userId);

    if (session.completed) {
      throw new ForbiddenException('Session already completed');
    }

    const updatedSession = await this.prisma.session.update({
      where: { id },
      data: {
        endTime: new Date(),
        completed: false, // Mark as not completed (skipped)
        notes: notes || 'Session skipped',
        quality: 1, // Low quality for skipped sessions
      },
      include: {
        user: {
          select: { id: true, firstName: true, lastName: true, email: true, avatar: true },
        },
        task: {
          select: { id: true, title: true },
        },
      },
    });

    // Note: We do NOT update task progress for skipped sessions
    // This is different from completed sessions which may increment task progress

    return updatedSession;
  }
}