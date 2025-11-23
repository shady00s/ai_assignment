import { Injectable, NotFoundException } from '@nestjs/common';
import { DatabaseService } from '../config/database.config';
import { CreateNotificationDto } from './dto/create-notification.dto';
import { NotificationType } from './dto/create-notification.dto';

export interface Notification {
  id: string;
  userId: string;
  type: NotificationType;
  title: string;
  message: string;
  read: boolean;
  entityId?: string;
  entityType?: string;
  data?: Record<string, any>;
  createdAt: string;
  readAt?: string;
  scheduledFor?: string;
}

@Injectable()
export class NotificationsService {
  constructor(private readonly prisma: DatabaseService) {}

  async create(createNotificationDto: CreateNotificationDto, userId: string): Promise<Notification> {
    const notification = await this.prisma.notification.create({
      data: {
        userId,
        type: createNotificationDto.type,
        title: createNotificationDto.title,
        message: createNotificationDto.message,
        entityId: createNotificationDto.entityId,
        entityType: createNotificationDto.entityType,
        data: createNotificationDto.data ? JSON.stringify(createNotificationDto.data) : null,
        scheduledFor: createNotificationDto.scheduledFor ? new Date(createNotificationDto.scheduledFor) : null,
      },
    });

    return {
      id: notification.id,
      userId: notification.userId,
      type: notification.type as NotificationType,
      title: notification.title,
      message: notification.message,
      read: notification.read,
      entityId: notification.entityId || undefined,
      entityType: notification.entityType || undefined,
      data: notification.data ? JSON.parse(notification.data) : undefined,
      createdAt: notification.createdAt.toISOString(),
      readAt: notification.readAt ? notification.readAt.toISOString() : undefined,
      scheduledFor: notification.scheduledFor ? notification.scheduledFor.toISOString() : undefined,
    };
  }

  async findAll(userId: string, options?: {
    unreadOnly?: boolean;
    type?: NotificationType;
    limit?: number;
    offset?: number;
  }): Promise<{ notifications: Notification[]; total: number }> {
    const { unreadOnly = false, type, limit = 50, offset = 0 } = options || {};

    const whereClause: any = { userId };

    if (unreadOnly) {
      whereClause.read = false;
    }

    if (type) {
      whereClause.type = type;
    }

    const [notifications, total] = await Promise.all([
      this.prisma.notification.findMany({
        where: whereClause,
        orderBy: [
          { scheduledFor: { sort: 'desc', nulls: 'first' } },
          { createdAt: 'desc' },
        ],
        take: limit,
        skip: offset,
      }),
      this.prisma.notification.count({ where: whereClause }),
    ]);

    const formattedNotifications = notifications.map(notification => ({
      id: notification.id,
      userId: notification.userId,
      type: notification.type as NotificationType,
      title: notification.title,
      message: notification.message,
      read: notification.read,
      entityId: notification.entityId || undefined,
      entityType: notification.entityType || undefined,
      data: notification.data ? JSON.parse(notification.data) : undefined,
      createdAt: notification.createdAt.toISOString(),
      readAt: notification.readAt ? notification.readAt.toISOString() : undefined,
      scheduledFor: notification.scheduledFor ? notification.scheduledFor.toISOString() : undefined,
    }));

    return {
      notifications: formattedNotifications,
      total,
    };
  }

  async markAsRead(id: string, userId: string): Promise<Notification> {
    const notification = await this.prisma.notification.findFirst({
      where: { id, userId },
    });

    if (!notification) {
      throw new NotFoundException('Notification not found');
    }

    const updatedNotification = await this.prisma.notification.update({
      where: { id },
      data: {
        read: true,
        readAt: new Date(),
      },
    });

    return {
      id: updatedNotification.id,
      userId: updatedNotification.userId,
      type: updatedNotification.type as NotificationType,
      title: updatedNotification.title,
      message: updatedNotification.message,
      read: updatedNotification.read,
      entityId: updatedNotification.entityId || undefined,
      entityType: updatedNotification.entityType || undefined,
      data: updatedNotification.data ? JSON.parse(updatedNotification.data) : undefined,
      createdAt: updatedNotification.createdAt.toISOString(),
      readAt: updatedNotification.readAt ? updatedNotification.readAt.toISOString() : undefined,
      scheduledFor: updatedNotification.scheduledFor ? updatedNotification.scheduledFor.toISOString() : undefined,
    };
  }

  async markAllAsRead(userId: string): Promise<void> {
    await this.prisma.notification.updateMany({
      where: {
        userId,
        read: false,
      },
      data: {
        read: true,
        readAt: new Date(),
      },
    });
  }

  async delete(id: string, userId: string): Promise<void> {
    const notification = await this.prisma.notification.findFirst({
      where: { id, userId },
    });

    if (!notification) {
      throw new NotFoundException('Notification not found');
    }

    await this.prisma.notification.delete({
      where: { id },
    });
  }

  async getUnreadCount(userId: string): Promise<number> {
    return this.prisma.notification.count({
      where: {
        userId,
        read: false,
        OR: [
        { scheduledFor: null },
        { scheduledFor: { lte: new Date() } },
      ],
      },
    });
  }

  async createAchievementNotification(
    userId: string,
    achievementId: string,
    achievementTitle: string,
    description: string
  ): Promise<Notification> {
    return this.create({
      type: NotificationType.ACHIEVEMENT,
      title: 'Achievement Unlocked! 🎉',
      message: `You've earned "${achievementTitle}": ${description}`,
      entityId: achievementId,
      entityType: 'achievement',
    }, userId);
  }

  async createTaskAssignedNotification(
    userId: string,
    taskId: string,
    taskTitle: string,
    assignerName: string
  ): Promise<Notification> {
    return this.create({
      type: NotificationType.TASK_ASSIGNED,
      title: 'New Task Assigned',
      message: `${assignerName} assigned you to "${taskTitle}"`,
      entityId: taskId,
      entityType: 'task',
    }, userId);
  }

  async createDeadlineReminderNotification(
    userId: string,
    taskId: string,
    taskTitle: string,
    dueDate: Date
  ): Promise<Notification> {
    return this.create({
      type: NotificationType.DEADLINE_REMINDER,
      title: 'Deadline Approaching',
      message: `"${taskTitle}" is due on ${dueDate.toLocaleDateString()}`,
      entityId: taskId,
      entityType: 'task',
    }, userId);
  }

  async createStreakMilestoneNotification(
    userId: string,
    streakCount: number
  ): Promise<Notification> {
    return this.create({
      type: NotificationType.STREAK_MILESTONE,
      title: 'Streak Milestone! 🔥',
      message: `You've maintained a ${streakCount}-day productivity streak!`,
    }, userId);
  }

  async createLevelUpNotification(
    userId: string,
    newLevel: number
  ): Promise<Notification> {
    return this.create({
      type: NotificationType.LEVEL_UP,
      title: 'Level Up! ⬆️',
      message: `Congratulations! You've reached level ${newLevel}!`,
    }, userId);
  }
}