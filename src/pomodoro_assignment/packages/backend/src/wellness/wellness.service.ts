import { Injectable, NotFoundException, ForbiddenException, ConflictException } from '@nestjs/common';
import { DatabaseService } from '../config/database.config';
import {
  CreateWellnessEntryDto,
  UpdateWellnessEntryDto,
  CreateWellnessReminderDto,
  UpdateWellnessReminderDto,
  CreateWellnessGoalDto,
  UpdateWellnessGoalDto,
  WellnessHistoryQueryDto,
  WellnessAnalyticsQueryDto,
  WellnessReminderType,
  WellnessGoalCategory,
  WellnessGoalPeriod
} from './dto';

@Injectable()
export class WellnessService {
  constructor(private readonly prisma: DatabaseService) {}

  // ========================================
  // WELLNESS ENTRY OPERATIONS
  // ========================================

  /**
   * Get today's wellness entry for a user
   */
  async getTodayWellnessEntry(userId: string) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const entry = await this.prisma.wellnessEntry.findFirst({
      where: {
        userId,
        date: {
          gte: today,
          lt: new Date(today.getTime() + 24 * 60 * 60 * 1000)
        }
      }
    });

    if (!entry) {
      return null;
    }

    return this.formatWellnessEntryResponse(entry);
  }

  /**
   * Create or update today's wellness entry
   */
  async createOrUpdateWellnessEntry(userId: string, data: CreateWellnessEntryDto) {
    const targetDate = data.date ? new Date(data.date) : new Date();
    targetDate.setHours(0, 0, 0, 0);

    const existingEntry = await this.prisma.wellnessEntry.findFirst({
      where: {
        userId,
        date: {
          gte: targetDate,
          lt: new Date(targetDate.getTime() + 24 * 60 * 60 * 1000)
        }
      }
    });

    if (existingEntry) {
      const updatedEntry = await this.prisma.wellnessEntry.update({
        where: { id: existingEntry.id },
        data: {
          ...data,
          date: targetDate,
        }
      });
      return this.formatWellnessEntryResponse(updatedEntry);
    } else {
      const newEntry = await this.prisma.wellnessEntry.create({
        data: {
          userId,
          ...data,
          date: targetDate,
        }
      });
      return this.formatWellnessEntryResponse(newEntry);
    }
  }

  /**
   * Update a specific wellness entry by date
   */
  async updateWellnessEntryByDate(userId: string, date: Date, data: UpdateWellnessEntryDto) {
    const targetDate = new Date(date);
    targetDate.setHours(0, 0, 0, 0);

    const entry = await this.prisma.wellnessEntry.findFirst({
      where: {
        userId,
        date: {
          gte: targetDate,
          lt: new Date(targetDate.getTime() + 24 * 60 * 60 * 1000)
        }
      }
    });

    if (!entry) {
      throw new NotFoundException('Wellness entry not found for the specified date');
    }

    const updatedEntry = await this.prisma.wellnessEntry.update({
      where: { id: entry.id },
      data
    });

    return this.formatWellnessEntryResponse(updatedEntry);
  }

  /**
   * Delete a wellness entry by date
   */
  async deleteWellnessEntryByDate(userId: string, date: Date) {
    const targetDate = new Date(date);
    targetDate.setHours(0, 0, 0, 0);

    const entry = await this.prisma.wellnessEntry.findFirst({
      where: {
        userId,
        date: {
          gte: targetDate,
          lt: new Date(targetDate.getTime() + 24 * 60 * 60 * 1000)
        }
      }
    });

    if (!entry) {
      throw new NotFoundException('Wellness entry not found for the specified date');
    }

    await this.prisma.wellnessEntry.delete({
      where: { id: entry.id }
    });

    return { message: 'Wellness entry deleted successfully' };
  }

  /**
   * Get wellness history for a user
   */
  async getWellnessHistory(userId: string, query: WellnessHistoryQueryDto) {
    const {
      startDate,
      endDate,
      days = 30,
      page = 1,
      limit = 10,
      sortBy = 'date',
      sortOrder = 'desc'
    } = query;

    let dateFilter: any = {};

    if (startDate || endDate) {
      if (startDate) {
        const start = new Date(startDate);
        start.setHours(0, 0, 0, 0);
        dateFilter.gte = start;
      }
      if (endDate) {
        const end = new Date(endDate);
        end.setHours(23, 59, 59, 999);
        dateFilter.lte = end;
      }
    } else {
      const end = new Date();
      end.setHours(23, 59, 59, 999);
      const start = new Date(end);
      start.setDate(start.getDate() - days);
      start.setHours(0, 0, 0, 0);
      dateFilter = { gte: start, lte: end };
    }

    const skip = (page - 1) * limit;

    const [entries, total] = await Promise.all([
      this.prisma.wellnessEntry.findMany({
        where: {
          userId,
          date: dateFilter
        },
        orderBy: {
          [sortBy]: sortOrder
        },
        skip,
        take: limit
      }),
      this.prisma.wellnessEntry.count({
        where: {
          userId,
          date: dateFilter
        }
      })
    ]);

    return {
      entries: entries.map(entry => this.formatWellnessEntryResponse(entry)),
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
        hasNext: page * limit < total,
        hasPrevious: page > 1
      }
    };
  }

  // ========================================
  // WELLNESS REMINDER OPERATIONS
  // ========================================

  /**
   * Get all wellness reminders for a user
   */
  async getWellnessReminders(userId: string) {
    const reminders = await this.prisma.wellnessReminder.findMany({
      where: { userId },
      orderBy: { type: 'asc' }
    });

    return reminders.map(reminder => ({
      ...reminder,
      weekdays: JSON.parse(reminder.weekdays)
    }));
  }

  /**
   * Create a new wellness reminder
   */
  async createWellnessReminder(userId: string, data: CreateWellnessReminderDto) {
    const existingReminder = await this.prisma.wellnessReminder.findFirst({
      where: {
        userId,
        type: data.type
      }
    });

    if (existingReminder) {
      throw new ConflictException(`A reminder of type ${data.type} already exists for this user`);
    }

    const reminder = await this.prisma.wellnessReminder.create({
      data: {
        userId,
        ...data,
        weekdays: JSON.stringify(data.weekdays)
      }
    });

    return {
      ...reminder,
      weekdays: JSON.parse(reminder.weekdays)
    };
  }

  /**
   * Update a wellness reminder
   */
  async updateWellnessReminder(userId: string, reminderId: string, data: UpdateWellnessReminderDto) {
    const reminder = await this.prisma.wellnessReminder.findFirst({
      where: { userId, id: reminderId }
    });

    if (!reminder) {
      throw new NotFoundException('Wellness reminder not found');
    }

    const updateData: any = { ...data };
    if (data.weekdays) {
      updateData.weekdays = JSON.stringify(data.weekdays);
    }

    const updatedReminder = await this.prisma.wellnessReminder.update({
      where: { id: reminderId },
      data: updateData
    });

    return {
      ...updatedReminder,
      weekdays: JSON.parse(updatedReminder.weekdays)
    };
  }

  /**
   * Delete a wellness reminder
   */
  async deleteWellnessReminder(userId: string, reminderId: string) {
    const reminder = await this.prisma.wellnessReminder.findFirst({
      where: { userId, id: reminderId }
    });

    if (!reminder) {
      throw new NotFoundException('Wellness reminder not found');
    }

    await this.prisma.wellnessReminder.delete({
      where: { id: reminderId }
    });

    return { message: 'Wellness reminder deleted successfully' };
  }

  // ========================================
  // WELLNESS GOAL OPERATIONS
  // ========================================

  /**
   * Get all wellness goals for a user
   */
  async getWellnessGoals(userId: string) {
    const goals = await this.prisma.wellnessGoal.findMany({
      where: { userId },
      orderBy: { category: 'asc' }
    });

    return Promise.all(goals.map(goal => this.formatWellnessGoalWithProgress(userId, goal)));
  }

  /**
   * Create a new wellness goal
   */
  async createWellnessGoal(userId: string, data: CreateWellnessGoalDto) {
    const existingGoal = await this.prisma.wellnessGoal.findFirst({
      where: {
        userId,
        category: data.category,
        period: data.period
      }
    });

    if (existingGoal) {
      throw new ConflictException(`A ${data.category.toLowerCase()} goal for ${data.period.toLowerCase()} period already exists`);
    }

    const goal = await this.prisma.wellnessGoal.create({
      data: { userId, ...data }
    });

    return this.formatWellnessGoalWithProgress(userId, goal);
  }

  /**
   * Update a wellness goal
   */
  async updateWellnessGoal(userId: string, goalId: string, data: UpdateWellnessGoalDto) {
    const goal = await this.prisma.wellnessGoal.findFirst({
      where: { userId, id: goalId }
    });

    if (!goal) {
      throw new NotFoundException('Wellness goal not found');
    }

    const updatedGoal = await this.prisma.wellnessGoal.update({
      where: { id: goalId },
      data
    });

    return this.formatWellnessGoalWithProgress(userId, updatedGoal);
  }

  /**
   * Delete a wellness goal
   */
  async deleteWellnessGoal(userId: string, goalId: string) {
    const goal = await this.prisma.wellnessGoal.findFirst({
      where: { userId, id: goalId }
    });

    if (!goal) {
      throw new NotFoundException('Wellness goal not found');
    }

    await this.prisma.wellnessGoal.delete({
      where: { id: goalId }
    });

    return { message: 'Wellness goal deleted successfully' };
  }

  // ========================================
  // WELLNESS ANALYTICS
  // ========================================

  /**
   * Get comprehensive wellness analytics
   */
  async getWellnessAnalytics(userId: string, query: WellnessAnalyticsQueryDto) {
    const {
      days = 30,
      startDate,
      endDate,
      category = 'ALL',
      includeRecommendations = true,
      includeTrends = false
    } = query;

    let analysisStart: Date;
    let analysisEnd: Date;

    if (startDate && endDate) {
      analysisStart = new Date(startDate);
      analysisEnd = new Date(endDate);
    } else {
      analysisEnd = new Date();
      analysisStart = new Date(analysisEnd);
      analysisStart.setDate(analysisStart.getDate() - days);
    }

    analysisStart.setHours(0, 0, 0, 0);
    analysisEnd.setHours(23, 59, 59, 999);

    const entries = await this.prisma.wellnessEntry.findMany({
      where: {
        userId,
        date: {
          gte: analysisStart,
          lte: analysisEnd
        }
      },
      orderBy: { date: 'asc' }
    });

    const analytics = {
      userId,
      period: days,
      startDate: analysisStart,
      endDate: analysisEnd,
      hydration: this.analyzeHydrationPatterns(entries),
      movement: this.analyzeMovementPatterns(entries),
      mentalWellness: this.analyzeMentalWellnessPatterns(entries),
      sleep: this.analyzeSleepPatterns(entries),
      overall: this.calculateOverallWellness(entries)
    };

    const result: any = analytics;

    if (includeTrends) {
      result.trends = this.calculateTrends(entries);
    }

    if (includeRecommendations) {
      result.recommendations = await this.generateRecommendations(userId, entries);
    }

    return result;
  }

  /**
   * Get wellness summary (simplified version for dashboard)
   */
  async getWellnessSummary(userId: string) {
    const today = await this.getTodayWellnessEntry(userId);
    const weeklyEntries = await this.getRecentWellnessEntries(userId, 7);

    return {
      today: today || this.getDefaultWellnessEntry(),
      weeklyAverage: this.calculateWeeklyAverages(weeklyEntries),
      streak: await this.getWellnessStreak(userId),
      goals: await this.getActiveGoals(userId)
    };
  }

  // ========================================
  // PRIVATE HELPER METHODS
  // ========================================

  private formatWellnessEntryResponse(entry: any) {
    const hydrationProgress = entry.hydrationGoal > 0
      ? Math.min(100, (entry.hydrationGlasses / entry.hydrationGoal) * 100)
      : 0;

    const wellnessScore = this.calculateWellnessScore(entry);

    return {
      ...entry,
      hydrationProgress: Math.round(hydrationProgress),
      wellnessScore
    };
  }

  private calculateWellnessScore(entry: any): number {
    let score = 0;
    let factors = 0;

    // Hydration (25% of score)
    const hydrationScore = entry.hydrationGoal > 0
      ? Math.min(100, (entry.hydrationGlasses / entry.hydrationGoal) * 100)
      : 50;
    score += hydrationScore * 0.25;
    factors++;

    // Movement (20% of score)
    const movementScore = Math.min(100, (entry.movementBreaks / 5) * 100); // 5 breaks = 100%
    score += movementScore * 0.20;
    factors++;

    // Mental wellness (30% of score)
    const moodScore = (entry.moodRating / 5) * 100;
    const stressScore = ((6 - entry.stressLevel) / 5) * 100; // Invert stress (lower is better)
    const energyScore = (entry.energyLevel / 5) * 100;
    const mentalScore = (moodScore + stressScore + energyScore) / 3;
    score += mentalScore * 0.30;
    factors++;

    // Mindfulness (15% of score)
    const mindfulnessScore = Math.min(100, (entry.meditationMinutes / 15) * 100);
    score += mindfulnessScore * 0.15;
    factors++;

    // Posture & Eye rest (10% of score)
    const postureScore = Math.min(100, (entry.postureChecks / 6) * 100);
    const eyeRestScore = Math.min(100, (entry.eyeRestBreaks / 4) * 100);
    const postureEyeScore = (postureScore + eyeRestScore) / 2;
    score += postureEyeScore * 0.10;
    factors++;

    return Math.round(score);
  }

  private async formatWellnessGoalWithProgress(userId: string, goal: any) {
    const currentProgress = await this.getCurrentGoalProgress(userId, goal);
    const progressPercentage = goal.targetValue > 0
      ? Math.min(100, (currentProgress / goal.targetValue) * 100)
      : 0;

    return {
      ...goal,
      currentProgress,
      progressPercentage: Math.round(progressPercentage)
    };
  }

  private async getCurrentGoalProgress(userId: string, goal: any): Promise<number> {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    let startDate = new Date(today);

    switch (goal.period) {
      case 'DAILY':
        startDate = today;
        break;
      case 'WEEKLY':
        startDate.setDate(startDate.getDate() - 7);
        break;
      case 'MONTHLY':
        startDate.setMonth(startDate.getMonth() - 1);
        break;
    }

    const entries = await this.prisma.wellnessEntry.findMany({
      where: {
        userId,
        date: {
          gte: startDate,
          lte: new Date()
        }
      }
    });

    switch (goal.category) {
      case 'HYDRATION':
        return entries.reduce((sum, entry) => sum + entry.hydrationGlasses, 0);
      case 'MOVEMENT':
        return entries.reduce((sum, entry) => sum + entry.movementBreaks, 0);
      case 'MEDITATION':
        return entries.reduce((sum, entry) => sum + entry.meditationMinutes, 0);
      case 'SLEEP':
        const sleepEntries = entries.filter(e => e.sleepHours);
        return sleepEntries.length > 0
          ? sleepEntries.reduce((sum, entry) => sum + entry.sleepHours!, 0) / sleepEntries.length
          : 0;
      default:
        return 0;
    }
  }

  private analyzeHydrationPatterns(entries: any[]) {
    const dailyAverages: { [key: number]: number } = {};
    entries.forEach(entry => {
      const day = entry.date.getDay();
      dailyAverages[day] = (dailyAverages[day] || 0) + entry.hydrationGlasses;
    });

    const weeklyAverage = Object.keys(dailyAverages).length > 0
      ? Object.values(dailyAverages).reduce((a, b) => a + b, 0) / Object.keys(dailyAverages).length
      : 0;

    const bestDayEntry = Object.entries(dailyAverages).sort(([, a], [, b]) => b - a)[0];
    const bestDay = parseInt(bestDayEntry?.[0] || '0');
    const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

    return {
      weeklyAverage: Math.round(weeklyAverage * 10) / 10,
      bestDay: dayNames[bestDay],
      consistencyScore: this.calculateConsistency(dailyAverages),
      trend: this.calculateTrend(dailyAverages),
      goalAchievementRate: this.calculateGoalAchievementRate(entries, 'hydrationGlasses', 'hydrationGoal')
    };
  }

  private analyzeMovementPatterns(entries: any[]) {
    const totalBreaks = entries.reduce((sum, entry) => sum + entry.movementBreaks, 0);
    const totalMinutes = entries.reduce((sum, entry) => sum + entry.movementMinutes, 0);

    return {
      averageBreaks: entries.length > 0 ? totalBreaks / entries.length : 0,
      averageMinutes: entries.length > 0 ? totalMinutes / entries.length : 0,
      weeklyTotal: totalBreaks,
      goalAchievementRate: this.calculateGoalAchievementRate(entries, 'movementBreaks', 5) // 5 breaks is goal
    };
  }

  private analyzeMentalWellnessPatterns(entries: any[]) {
    return {
      averageMoodRating: entries.reduce((sum, e) => sum + e.moodRating, 0) / entries.length || 0,
      averageStressLevel: entries.reduce((sum, e) => sum + e.stressLevel, 0) / entries.length || 0,
      averageEnergyLevel: entries.reduce((sum, e) => sum + e.energyLevel, 0) / entries.length || 0,
      meditationStreak: this.calculateMeditationStreak(entries),
      totalMindfulnessSessions: entries.reduce((sum, e) => sum + e.mindfulnessSessions, 0)
    };
  }

  private analyzeSleepPatterns(entries: any[]) {
    const sleepEntries = entries.filter(e => e.sleepHours && e.sleepQuality);

    if (sleepEntries.length === 0) {
      return {
        averageHours: 0,
        averageQuality: 0,
        consistencyScore: 0,
        bestSleepDay: 'N/A'
      };
    }

    return {
      averageHours: sleepEntries.reduce((sum, e) => sum + e.sleepHours!, 0) / sleepEntries.length,
      averageQuality: sleepEntries.reduce((sum, e) => sum + e.sleepQuality!, 0) / sleepEntries.length,
      consistencyScore: this.calculateSleepConsistency(sleepEntries),
      bestSleepDay: this.findBestSleepDay(sleepEntries)
    };
  }

  private calculateOverallWellness(entries: any[]) {
    const scores = entries.map(entry => this.calculateWellnessScore(entry));
    const averageScore = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;

    return {
      overallScore: Math.round(averageScore),
      trendDirection: this.calculateScoreTrend(scores),
      streakDays: this.calculateWellnessStreak(entries),
      perfectDaysCount: this.countPerfectDays(entries),
      complianceRate: this.calculateComplianceRate(entries)
    };
  }

  private calculateConsistency(values: { [key: number]: number }): number {
    const vals = Object.values(values);
    if (vals.length <= 1) return 0;

    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance = vals.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / vals.length;
    const standardDeviation = Math.sqrt(variance);

    return Math.max(0, 1 - (standardDeviation / mean));
  }

  private calculateTrend(values: { [key: number]: number }): 'improving' | 'stable' | 'declining' {
    const vals = Object.entries(values).sort(([a], [b]) => parseInt(a) - parseInt(b));
    if (vals.length < 3) return 'stable';

    const firstHalf = vals.slice(0, Math.floor(vals.length / 2));
    const secondHalf = vals.slice(Math.floor(vals.length / 2));

    const firstAvg = firstHalf.reduce((sum, [, val]) => sum + val, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((sum, [, val]) => sum + val, 0) / secondHalf.length;

    const change = (secondAvg - firstAvg) / firstAvg;

    if (change > 0.1) return 'improving';
    if (change < -0.1) return 'declining';
    return 'stable';
  }

  private calculateGoalAchievementRate(entries: any[], valueField: string, goalField: string | number): number {
    if (typeof goalField === 'number') {
      const achieved = entries.filter(entry => (entry as any)[valueField] >= goalField).length;
      return entries.length > 0 ? achieved / entries.length : 0;
    } else {
      const achieved = entries.filter(entry => (entry as any)[valueField] >= (entry as any)[goalField]).length;
      return entries.length > 0 ? achieved / entries.length : 0;
    }
  }

  private calculateMeditationStreak(entries: any[]): number {
    let streak = 0;
    const sortedEntries = entries.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

    for (const entry of sortedEntries) {
      if (entry.meditationMinutes > 0) {
        streak++;
      } else {
        break;
      }
    }

    return streak;
  }

  private calculateSleepConsistency(sleepEntries: any[]): number {
    const hours = sleepEntries.map(e => e.sleepHours!);
    return this.calculateConsistencyFromValues(hours);
  }

  private findBestSleepDay(sleepEntries: any[]): string {
    const dayScores: { [key: number]: number[] } = {};

    sleepEntries.forEach(entry => {
      const day = entry.date.getDay();
      if (!dayScores[day]) dayScores[day] = [];
      dayScores[day].push(entry.sleepQuality!);
    });

    const dayAverages = Object.entries(dayScores).map(([day, qualities]) => ({
      day: parseInt(day),
      avg: qualities.reduce((a, b) => a + b, 0) / qualities.length
    }));

    const bestDay = dayAverages.sort((a, b) => b.avg - a.avg)[0]?.day || 0;
    const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

    return dayNames[bestDay];
  }

  private calculateConsistencyFromValues(values: number[]): number {
    if (values.length <= 1) return 0;

    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const standardDeviation = Math.sqrt(variance);

    return Math.max(0, 1 - (standardDeviation / mean));
  }

  private calculateScoreTrend(scores: number[]): 'upward' | 'stable' | 'downward' {
    if (scores.length < 3) return 'stable';

    const firstHalf = scores.slice(0, Math.floor(scores.length / 2));
    const secondHalf = scores.slice(Math.floor(scores.length / 2));

    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

    const change = (secondAvg - firstAvg) / firstAvg;

    if (change > 0.05) return 'upward';
    if (change < -0.05) return 'downward';
    return 'stable';
  }

  private calculateWellnessStreak(entries: any[]): number {
    // Implementation would check consecutive days with wellness score > 70
    return Math.floor(Math.random() * 7); // Placeholder
  }

  private countPerfectDays(entries: any[]): number {
    return entries.filter(entry => this.calculateWellnessScore(entry) >= 90).length;
  }

  private calculateComplianceRate(entries: any[]): number {
    const totalFields = entries.length * 5; // Assuming 5 key wellness metrics
    const completedFields = entries.reduce((sum, entry) => {
      const completed = [
        entry.hydrationGlasses > 0,
        entry.movementBreaks > 0,
        entry.moodRating > 0,
        entry.stressLevel > 0,
        entry.energyLevel > 0
      ].filter(Boolean).length;
      return sum + completed;
    }, 0);

    return totalFields > 0 ? completedFields / totalFields : 0;
  }

  private calculateTrends(entries: any[]) {
    return entries.slice(-7).map(entry => ({
      date: entry.date.toISOString().split('T')[0],
      hydrationGlasses: entry.hydrationGlasses,
      movementBreaks: entry.movementBreaks,
      moodRating: entry.moodRating,
      stressLevel: entry.stressLevel,
      energyLevel: entry.energyLevel,
      wellnessScore: this.calculateWellnessScore(entry)
    }));
  }

  private async generateRecommendations(userId: string, entries: any[]) {
    // Calculate patterns directly from entries to avoid infinite recursion
    const hydrationAverage = entries.length > 0
      ? entries.reduce((sum, e) => sum + e.hydrationGlasses, 0) / entries.length
      : 0;

    const movementAverage = entries.length > 0
      ? entries.reduce((sum, e) => sum + e.movementBreaks, 0) / entries.length
      : 0;

    const moodAverage = entries.length > 0
      ? entries.reduce((sum, e) => sum + e.moodRating, 0) / entries.length
      : 0;

    const recommendations = [];

    if (hydrationAverage < 6) {
      recommendations.push({
        id: 'wellness_rec_001',
        type: 'HYDRATION',
        title: 'Increase Your Water Intake',
        description: `You're averaging ${hydrationAverage.toFixed(1)} glasses/day. Try setting hourly reminders to reach your goal of 8 glasses!`,
        priority: 'HIGH',
        actionable: true,
        estimatedImpact: '+15 wellness score'
      });
    }

    if (movementAverage < 4) {
      recommendations.push({
        id: 'wellness_rec_002',
        type: 'MOVEMENT',
        title: 'Take More Movement Breaks',
        description: 'Research shows movement breaks every hour improve focus and health. Try standing and stretching every 60 minutes.',
        priority: 'MEDIUM',
        actionable: true,
        estimatedImpact: '+10 wellness score'
      });
    }

    if (moodAverage < 3.5) {
      recommendations.push({
        id: 'wellness_rec_003',
        type: 'MENTAL_WELLNESS',
        title: 'Boost Your Mood',
        description: 'Your mood has been lower than usual. Consider short meditation sessions or a brief walk outside.',
        priority: 'MEDIUM',
        actionable: true,
        estimatedImpact: '+12 wellness score'
      });
    }

    return recommendations;
  }

  private async getRecentWellnessEntries(userId: string, days: number) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    startDate.setHours(0, 0, 0, 0);

    return await this.prisma.wellnessEntry.findMany({
      where: {
        userId,
        date: {
          gte: startDate
        }
      },
      orderBy: { date: 'desc' }
    });
  }

  private calculateWeeklyAverages(entries: any[]) {
    if (entries.length === 0) {
      return this.getDefaultWellnessEntry();
    }

    return {
      hydrationGlasses: Math.round(entries.reduce((sum, e) => sum + e.hydrationGlasses, 0) / entries.length),
      movementBreaks: Math.round(entries.reduce((sum, e) => sum + e.movementBreaks, 0) / entries.length),
      moodRating: Math.round((entries.reduce((sum, e) => sum + e.moodRating, 0) / entries.length) * 10) / 10,
      stressLevel: Math.round((entries.reduce((sum, e) => sum + e.stressLevel, 0) / entries.length) * 10) / 10,
      energyLevel: Math.round((entries.reduce((sum, e) => sum + e.energyLevel, 0) / entries.length) * 10) / 10,
      meditationMinutes: Math.round(entries.reduce((sum, e) => sum + e.meditationMinutes, 0) / entries.length)
    };
  }

  private async getWellnessStreak(userId: string): Promise<number> {
    const entries = await this.getRecentWellnessEntries(userId, 30);
    let streak = 0;

    const sortedEntries = entries.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

    for (const entry of sortedEntries) {
      if (this.calculateWellnessScore(entry) >= 70) {
        streak++;
      } else {
        break;
      }
    }

    return streak;
  }

  private async getActiveGoals(userId: string) {
    const goals = await this.prisma.wellnessGoal.findMany({
      where: { userId, active: true }
    });

    return Promise.all(goals.map(goal => this.formatWellnessGoalWithProgress(userId, goal)));
  }

  private getDefaultWellnessEntry() {
    return {
      hydrationGlasses: 0,
      hydrationGoal: 8,
      movementBreaks: 0,
      movementMinutes: 0,
      moodRating: 3,
      stressLevel: 3,
      energyLevel: 3,
      meditationMinutes: 0,
      breathingExercises: 0,
      mindfulnessSessions: 0,
      postureChecks: 0,
      eyeRestBreaks: 0
    };
  }
}