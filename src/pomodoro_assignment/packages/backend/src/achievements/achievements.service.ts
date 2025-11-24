import { Injectable, NotFoundException, ForbiddenException, ConflictException } from '@nestjs/common';
import { DatabaseService } from '../config/database.config';
import { Achievement, UserAchievement, User } from '@prisma/client';

@Injectable()
export class AchievementsService {
  constructor(private readonly prisma: DatabaseService) {}

  /**
   * Get all available achievements
   */
  async getAllAchievements(): Promise<Achievement[]> {
    return this.prisma.achievement.findMany({
      where: {
        isActive: true,
      },
      orderBy: [
        { category: 'asc' },
        { createdAt: 'asc' },
      ],
    });
  }

  /**
   * Get user's achievements (both unlocked and in progress)
   */
  async getUserAchievements(userId: string): Promise<any[]> {
    const userAchievements = await this.prisma.userAchievement.findMany({
      where: { userId },
      include: {
        achievement: true,
      },
      orderBy: {
        unlockedAt: 'desc',
      },
    });

    // Convert progress from JSON string to number
    return userAchievements.map(ua => ({
      ...ua,
      progress: parseInt(ua.progress || '0'),
    }));
  }

  /**
   * Get a specific user achievement by ID
   */
  async getUserAchievementById(id: string, userId: string): Promise<any> {
    const userAchievement = await this.prisma.userAchievement.findUnique({
      where: { id },
      include: {
        achievement: true,
      },
    });

    if (!userAchievement) {
      throw new NotFoundException('User achievement not found');
    }

    if (userAchievement.userId !== userId) {
      throw new ForbiddenException('Access denied to this achievement');
    }

    // Convert progress from JSON string to number
    return {
      ...userAchievement,
      progress: parseInt(userAchievement.progress || '0'),
    };
  }

  /**
   * Check and update user progress on all achievements
   * This should be called after significant user actions (session completion, task completion, etc.)
   */
  async updateAchievementProgress(userId: string): Promise<any[]> {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        firstName: true,
        lastName: true,
        totalFocusTime: true,
        tasksCompleted: true,
        streak: true,
        level: true,
        xp: true,
        createdAt: true,
      },
    });

    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Get all active achievements
    const achievements = await this.prisma.achievement.findMany({
      where: { isActive: true },
    });

    const updatedAchievements: UserAchievement[] = [];

    for (const achievement of achievements) {
      try {
        const progress = await this.calculateAchievementProgress(userId, achievement, user);

        if (progress >= 100) {
          // Achievement is completed, unlock it if not already unlocked
          const unlockedAchievement = await this.unlockAchievement(userId, achievement.id);
          if (unlockedAchievement) {
            updatedAchievements.push(unlockedAchievement);
          }
        } else {
          // Update progress for in-progress achievement
          await this.updateOrCreateProgress(userId, achievement.id, progress);
        }
      } catch (error) {
        // Log error but continue with other achievements
        console.error(`Error updating achievement ${achievement.id}:`, error);
      }
    }

    return updatedAchievements;
  }

  /**
   * Manually unlock an achievement for a user
   */
  async unlockAchievement(userId: string, achievementId: string): Promise<any | null> {
    // Check if achievement exists and is active
    const achievement = await this.prisma.achievement.findUnique({
      where: { id: achievementId },
    });

    if (!achievement || !achievement.isActive) {
      throw new NotFoundException('Achievement not found or inactive');
    }

    // Check if user already has this achievement
    const existingAchievement = await this.prisma.userAchievement.findUnique({
      where: {
        userId_achievementId: {
          userId,
          achievementId,
        },
      },
    });

    if (existingAchievement) {
      return null; // Already unlocked
    }

    // Create the user achievement
    const userAchievement = await this.prisma.userAchievement.create({
      data: {
        userId,
        achievementId,
        progress: JSON.stringify(100), // Fully completed as JSON string
      },
      include: {
        achievement: true,
      },
    });

    // Award XP to user
    await this.prisma.user.update({
      where: { id: userId },
      data: {
        xp: {
          increment: achievement.xpValue,
        },
      },
    });

    // Return with progress as number for frontend
    return {
      ...userAchievement,
      progress: 100,
    };
  }

  /**
   * Calculate progress for a specific achievement
   */
  private async calculateAchievementProgress(
    userId: string,
    achievement: Achievement,
    user?: any,
  ): Promise<number> {
    if (!user) {
      user = await this.prisma.user.findUnique({
        where: { id: userId },
        select: {
          id: true,
          email: true,
          firstName: true,
          lastName: true,
          totalFocusTime: true,
          tasksCompleted: true,
          streak: true,
          level: true,
          xp: true,
          createdAt: true,
        },
      });

      if (!user) {
        throw new NotFoundException('User not found');
      }
    }

    const criteria = JSON.parse(achievement.criteria);
    let progress = 0;

    const now = new Date();
    let startDate: Date;

    // Calculate start date based on timeframe
    switch (criteria.timeframe) {
      case 'DAILY':
        startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        break;
      case 'WEEKLY':
        startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case 'MONTHLY':
        startDate = new Date(now.getFullYear(), now.getMonth(), 1);
        break;
      case 'ALL_TIME':
      default:
        startDate = user.createdAt;
        break;
    }

    switch (criteria.type) {
      case 'SESSION_COUNT':
        const sessionCount = await this.prisma.session.count({
          where: {
            userId,
            type: 'POMODORO',
            completed: true,
            startTime: {
              gte: startDate,
            },
          },
        });
        progress = Math.min(100, Math.round((sessionCount / criteria.value) * 100));
        break;

      case 'STREAK_DAYS':
        progress = Math.min(100, Math.round((user.streak / criteria.value) * 100));
        break;

      case 'TOTAL_TIME':
        progress = Math.min(100, Math.round((user.totalFocusTime / criteria.value) * 100));
        break;

      case 'TASKS_COMPLETED':
        progress = Math.min(100, Math.round((user.tasksCompleted / criteria.value) * 100));
        break;

      case 'TEAM_HELP':
        // Count tasks where user is assignee (helping team members)
        const helpedTasks = await this.prisma.task.count({
          where: {
            assigneeId: userId,
            creatorId: { not: userId }, // Not their own task
            status: 'COMPLETED',
            completedAt: {
              gte: startDate,
            },
          },
        });
        progress = Math.min(100, Math.round((helpedTasks / criteria.value) * 100));
        break;

      default:
        progress = 0;
        break;
    }

    return progress;
  }

  /**
   * Update or create progress entry for an achievement
   */
  private async updateOrCreateProgress(
    userId: string,
    achievementId: string,
    progress: number,
  ): Promise<void> {
    const existingProgress = await this.prisma.userAchievement.findUnique({
      where: {
        userId_achievementId: {
          userId,
          achievementId,
        },
      },
    });

    if (existingProgress) {
      const currentProgress = parseInt(existingProgress.progress || '0');
      await this.prisma.userAchievement.update({
        where: { id: existingProgress.id },
        data: {
          progress: JSON.stringify(Math.max(currentProgress, progress)),
        },
      });
    } else {
      // Create new progress entry (not unlocked yet)
      await this.prisma.userAchievement.create({
        data: {
          userId,
          achievementId,
          progress: JSON.stringify(progress),
        },
      });
    }
  }

  /**
   * Get achievement statistics for a user
   */
  async getUserAchievementStats(userId: string) {
    const [
      totalAchievements,
      unlockedAchievements,
      recentUnlocks,
    ] = await Promise.all([
      this.prisma.achievement.count({
        where: { isActive: true },
      }),
      this.prisma.userAchievement.count({
        where: { userId },
      }),
      this.prisma.userAchievement.findMany({
        where: { userId, progress: JSON.stringify(100) }, // Check for completed (100% progress)
        include: { achievement: true },
        orderBy: { unlockedAt: 'desc' },
        take: 5,
      }),
    ]);

    // Calculate total XP from achievements
    const unlockedWithAchievement = await this.prisma.userAchievement.findMany({
      where: { userId, progress: JSON.stringify(100) },
      include: { achievement: true },
    });

    const totalXpFromAchievements = unlockedWithAchievement.reduce(
      (total, ua) => total + (ua.achievement?.xpValue || 0),
      0,
    );

    const completionRate = totalAchievements > 0 ? (unlockedAchievements / totalAchievements) * 100 : 0;

    return {
      totalAchievements,
      unlockedAchievements,
      completionRate: Math.round(completionRate),
      totalXpFromAchievements,
      recentUnlocks,
    };
  }
}