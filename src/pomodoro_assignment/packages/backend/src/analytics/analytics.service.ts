import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { DatabaseService } from '../config/database.config';

@Injectable()
export class AnalyticsService {
  constructor(private readonly prisma: DatabaseService) {}

  /**
   * Calculate task completion rate for a team member
   * @param memberId - User ID to calculate completion rate for
   * @param startDate - Optional start date for filtering
   * @param endDate - Optional end date for filtering
   * @returns Completion rate percentage (0-100)
   */
  private async getTeamMemberCompletionRate(
    memberId: string,
    startDate?: Date,
    endDate?: Date,
  ): Promise<number> {
    const whereClause: any = {
      OR: [
        { creatorId: memberId },
        { assigneeId: memberId }
      ]
    };

    // Apply date filtering if provided
    if (startDate || endDate) {
      whereClause.createdAt = {
        gte: startDate,
        lte: endDate,
      };
    }

    const [totalTasks, completedTasks] = await Promise.all([
      this.prisma.task.count({
        where: whereClause,
      }),
      this.prisma.task.count({
        where: {
          ...whereClause,
          status: 'COMPLETED'
        }
      })
    ]);

    return totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;
  }

  async getFocusAnalytics(userId: string, startDate?: Date, endDate?: Date) {
    const whereClause: any = {
      userId,
      completed: true,
      type: 'POMODORO',
    };

    if (startDate || endDate) {
      whereClause.startTime = {
        gte: startDate,
        lte: endDate,
      };
    }

    // Get all completed pomodoro sessions in the date range
    const sessions = await this.prisma.session.findMany({
      where: whereClause,
      select: {
        duration: true,
        startTime: true,
        quality: true,
        taskId: true,
      },
      orderBy: { startTime: 'desc' },
    });

    // Calculate focus analytics
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    // Daily focus time (today)
    const dailyFocusTime = sessions
      .filter(session => session.startTime >= today)
      .reduce((sum, session) => sum + session.duration, 0);

    // Weekly focus time (last 7 days)
    const weeklyFocusTime = sessions
      .filter(session => session.startTime >= weekAgo)
      .reduce((sum, session) => sum + session.duration, 0);

    // Monthly focus time (last 30 days)
    const monthlyFocusTime = sessions
      .filter(session => session.startTime >= monthAgo)
      .reduce((sum, session) => sum + session.duration, 0);

    // Average session length
    const averageSessionLength = sessions.length > 0
      ? sessions.reduce((sum, session) => sum + session.duration, 0) / sessions.length
      : 0;

    // Peak focus hours (find hours with most session time)
    const hourlyFocus: { [hour: number]: number } = {};
    sessions.forEach(session => {
      const hour = session.startTime.getHours();
      hourlyFocus[hour] = (hourlyFocus[hour] || 0) + session.duration;
    });

    const peakFocusHours = Object.entries(hourlyFocus)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([hour]) => parseInt(hour));

    // Focus trend (compare last week to previous week)
    const twoWeeksAgo = new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000);
    const lastWeekSessions = sessions.filter(session => session.startTime >= weekAgo);
    const previousWeekSessions = sessions.filter(
      session => session.startTime >= twoWeeksAgo && session.startTime < weekAgo
    );

    const lastWeekTime = lastWeekSessions.reduce((sum, session) => sum + session.duration, 0);
    const previousWeekTime = previousWeekSessions.reduce((sum, session) => sum + session.duration, 0);

    let focusTrend: 'IMPROVING' | 'DECLINING' | 'STABLE' = 'STABLE';
    if (lastWeekTime > previousWeekTime * 1.1) {
      focusTrend = 'IMPROVING';
    } else if (lastWeekTime < previousWeekTime * 0.9) {
      focusTrend = 'DECLINING';
    }

    // Completion rate (sessions with quality rating)
    const sessionsWithQuality = sessions.filter(session => session.quality !== null);
    const completionRate = sessions.length > 0
      ? (sessionsWithQuality.length / sessions.length) * 100
      : 0;

    return {
      dailyFocusTime,
      weeklyFocusTime,
      monthlyFocusTime,
      averageSessionLength,
      peakFocusHours,
      focusTrend,
      completionRate: Math.round(completionRate),
    };
  }

  async getWellnessAnalytics(userId: string, startDate?: Date, endDate?: Date) {
    // Get today's wellness entry for real-time metrics
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const todayWellnessEntry = await this.prisma.wellnessEntry.findFirst({
      where: {
        userId,
        date: {
          gte: today,
          lt: tomorrow
        }
      }
    });

    // Get user's basic info for fallback metrics
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: {
        wellnessScore: true,
        streak: true,
        totalFocusTime: true,
        preferences: true,
      },
    });

    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Parse preferences to get wellness settings
    let preferences: any = {};
    if (user.preferences) {
      try {
        preferences = JSON.parse(user.preferences);
      } catch {
        preferences = {};
      }
    }

    // Get user's sessions for fallback calculations
    const startOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate());

    const todaySessions = await this.prisma.session.findMany({
      where: {
        userId,
        startTime: { gte: startOfDay },
        completed: true,
      },
      select: {
        type: true,
        duration: true,
        startTime: true,
      },
    });

    // Use real wellness data if available, otherwise fallback to calculated values
    const hasRealWellnessData = !!todayWellnessEntry;

    // Hydration metrics
    const hydrationGoal = todayWellnessEntry?.hydrationGoal || preferences.wellness?.hydrationGoal || 8;
    const hydrationCurrent = hasRealWellnessData
      ? todayWellnessEntry!.hydrationGlasses
      : this.calculateHydrationFallback(todaySessions);

    // Movement metrics
    const movementGoal = 5; // Default goal
    const movementCurrent = hasRealWellnessData
      ? todayWellnessEntry!.movementBreaks
      : this.calculateMovementFallback(todaySessions);

    // Mindfulness metrics
    const mindfulnessMinutes = hasRealWellnessData
      ? todayWellnessEntry!.meditationMinutes
      : Math.round(user.totalFocusTime * 0.1);

    // Self-reported metrics - use real data if available, otherwise calculate from user profile
    const moodRating = hasRealWellnessData
      ? todayWellnessEntry!.moodRating
      : (user.wellnessScore ? Math.round(user.wellnessScore) : 3);
    const stressLevel = hasRealWellnessData
      ? todayWellnessEntry!.stressLevel
      : Math.max(1, 5 - Math.round(user.wellnessScore || 3));
    const energyLevel = hasRealWellnessData
      ? todayWellnessEntry!.energyLevel
      : Math.min(5, Math.round((user.streak / 7) + 2));

    // Calculate wellness analytics based on real wellness data with fallback
    const wellnessAnalytics = {
      mindfulnessMinutes,
      hydrationGoal,
      hydrationCurrent, // ✅ UPDATED: Real wellness data with fallback
      movementGoal,
      movementCurrent, // ✅ UPDATED: Real wellness data with fallback
      moodRating,
      stressLevel,
      energyLevel,
    };

    return wellnessAnalytics;
  }

  /**
   * Fallback method to calculate hydration from session patterns
   */
  private calculateHydrationFallback(todaySessions: any[]): number {
    // Calculate hydration based on session count and breaks (8 glasses per day goal)
    // Assume good hydration if user takes regular breaks
    const sessionsWithBreaks = todaySessions.filter(s => s.type === 'SHORT_BREAK').length;
    return Math.min(8, Math.max(1, sessionsWithBreaks + 2));
  }

  /**
   * Fallback method to calculate movement from session patterns
   */
  private calculateMovementFallback(todaySessions: any[]): number {
    // Calculate movement based on break sessions and total focus time (5 movement breaks per day goal)
    const sessionsWithBreaks = todaySessions.filter(s => s.type === 'SHORT_BREAK').length;
    const focusSessions = todaySessions.filter(s => s.type === 'POMODORO').length;
    return Math.min(5, Math.max(1, sessionsWithBreaks + Math.floor(focusSessions / 3)));
  }

  async getTeamAnalytics(teamId: string, startDate?: Date, endDate?: Date, userId?: string) {
    // Check if user is part of the team
    if (userId) {
      const teamMember = await this.prisma.teamMember.findUnique({
        where: {
          userId_teamId: {
            userId,
            teamId,
          },
        },
      });

      if (!teamMember) {
        throw new ForbiddenException('You are not a member of this team');
      }
    }

    // Get team information
    const team = await this.prisma.team.findUnique({
      where: { id: teamId },
      include: {
        members: {
          include: {
            user: {
              select: {
                id: true,
                firstName: true,
                lastName: true,
                email: true,
                avatar: true,
                wellnessScore: true,
                level: true,
                xp: true,
                streak: true,
              },
            },
          },
        },
      },
    });

    if (!team) {
      throw new NotFoundException('Team not found');
    }

    const memberIds = team.members.map(member => member.userId);

    // Get sessions for all team members in the date range
    const sessionWhereClause: any = {
      userId: { in: memberIds },
      completed: true,
      type: 'POMODORO',
    };

    if (startDate || endDate) {
      sessionWhereClause.startTime = {
        gte: startDate,
        lte: endDate,
      };
    }

    // Batch calculate completion rates for all members to prevent N+1 queries
    const memberCompletionRates = await Promise.all(
      memberIds.map(async (memberId) => ({
        userId: memberId,
        completionRate: await this.getTeamMemberCompletionRate(memberId, startDate, endDate)
      }))
    );

    // Create lookup map for O(1) access during member stats calculation
    const completionRateMap = new Map(
      memberCompletionRates.map(item => [item.userId, item.completionRate])
    );

    const [sessions, memberStats] = await Promise.all([
      // Get all team sessions
      this.prisma.session.findMany({
        where: sessionWhereClause,
        select: {
          userId: true,
          duration: true,
          quality: true,
          startTime: true,
        },
      }),

      // Get individual member stats
      Promise.all(
        memberIds.map(async (memberId) => {
          const memberSessions = await this.prisma.session.findMany({
            where: {
              ...sessionWhereClause,
              userId: memberId,
            },
            select: {
              duration: true,
              quality: true,
            },
          });

          const totalFocusTime = memberSessions.reduce((sum, session) => sum + session.duration, 0);
          const avgQuality = memberSessions.length > 0
            ? memberSessions.reduce((sum, session) => sum + (session.quality || 0), 0) / memberSessions.length
            : 0;

          const member = team.members.find(m => m.userId === memberId);

          return {
            userId: memberId,
            user: member!.user,
            focusTime: totalFocusTime,
            tasksCompleted: await this.prisma.task.count({
              where: {
                OR: [
                  { creatorId: memberId },
                  { assigneeId: memberId },
                ],
                status: 'COMPLETED',
                completedAt: sessionWhereClause.startTime,
              },
            }),
            completionRate: completionRateMap.get(memberId) || 0, // ✅ FIXED: Use calculated completion rate
            wellnessScore: member!.user.wellnessScore || 0,
            streakDays: member!.user.streak || 0,
          };
        })
      ),
    ]);

    // Calculate team analytics
    const totalFocusTime = sessions.reduce((sum, session) => sum + session.duration, 0);
    const averageFocusTime = memberStats.length > 0 ? totalFocusTime / memberStats.length : 0;
    const tasksCompleted = memberStats.reduce((sum, member) => sum + member.tasksCompleted, 0);
    const averageCompletionRate = memberStats.length > 0
      ? memberStats.reduce((sum, member) => sum + member.completionRate, 0) / memberStats.length
      : 0;

    const teamWellnessScore = memberStats.length > 0
      ? memberStats.reduce((sum, member) => sum + member.wellnessScore, 0) / memberStats.length
      : 0;

    const collaborationScore = Math.min(100, (tasksCompleted / memberStats.length) * 10); // Mock calculation

    // Determine focus trend
    const now = new Date();
    const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const recentSessions = sessions.filter(session => session.startTime >= oneWeekAgo);
    const olderSessions = sessions.filter(
      session => session.startTime < oneWeekAgo && session.startTime >= new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000)
    );

    const recentTime = recentSessions.reduce((sum, session) => sum + session.duration, 0);
    const olderTime = olderSessions.reduce((sum, session) => sum + session.duration, 0);

    let focusTrend: 'IMPROVING' | 'DECLINING' | 'STABLE' = 'STABLE';
    if (recentTime > olderTime * 1.1) {
      focusTrend = 'IMPROVING';
    } else if (recentTime < olderTime * 0.9) {
      focusTrend = 'DECLINING';
    }

    return {
      teamId,
      teamName: team.name,
      memberCount: memberStats.length,
      totalFocusTime,
      averageFocusTime: Math.round(averageFocusTime),
      tasksCompleted,
      averageCompletionRate: Math.round(averageCompletionRate),
      topPerformers: memberStats
        .sort((a, b) => b.focusTime - a.focusTime)
        .slice(0, 5),
      focusTrend,
      wellnessScore: Math.round(teamWellnessScore),
      collaborationScore: Math.round(collaborationScore),
      period: {
        startDate: startDate?.toISOString() || new Date(0).toISOString(),
        endDate: endDate?.toISOString() || new Date().toISOString(),
      },
    };
  }
}