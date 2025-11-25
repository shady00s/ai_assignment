import {
  Injectable,
  NotFoundException,
  ConflictException,
  BadRequestException,
} from '@nestjs/common';
import { DatabaseService } from '../config/database.config';
import { LoggerService } from '../core/logger/logger.service';
import { CreateUserDto, UpdateUserDto, UpdatePreferencesDto } from './dto';
import * as bcrypt from 'bcryptjs';

@Injectable()
export class UsersService {
  constructor(
    private readonly databaseService: DatabaseService,
    private readonly logger: LoggerService,
  ) {}

  async create(createUserDto: CreateUserDto): Promise<any> {
    try {
      // Check if user with email already exists
      const existingUser = await this.databaseService.user.findUnique({
        where: { email: createUserDto.email },
      });

      if (existingUser) {
        throw new ConflictException('User with this email already exists');
      }

      // Don't set default preferences for new users - they should go through onboarding
      // Only set preferences if explicitly provided in the request
      const preferencesToSet = createUserDto.preferences || null;

      // Generate a random password for users created via this service
      const tempPassword = 'temp-password-' + Math.random().toString(36).slice(-8);
      const passwordHash = await bcrypt.hash(tempPassword, 12);

      const user = await this.databaseService.user.create({
        data: {
          email: createUserDto.email,
          firstName: createUserDto.firstName,
          lastName: createUserDto.lastName,
          password: passwordHash,
          avatar: createUserDto.avatar || null,
          level: createUserDto.level || 1,
          xp: createUserDto.xp || 0,
          preferences: preferencesToSet ? JSON.stringify(preferencesToSet) : null,
          teamId: createUserDto.teamId || null,
        },
        select: {
          id: true,
          email: true,
          password:true,
          firstName: true,
          lastName: true,
          avatar: true,
          level: true,
          xp: true,
          streak: true,
          totalFocusTime: true,
          tasksCompleted: true,
          qualityScore: true,
          wellnessScore: true,
          teamId: true,
          preferences: true,
          createdAt: true,
          updatedAt: true,
        },
      });

      this.logger.logUserAction('USER_CREATED', user.id, {
        email: user.email,
        teamId: user.teamId,
      });

      return {
        ...user,
        preferences: user.preferences ? JSON.parse(user.preferences as string) : null,
      };
    } catch (error) {
      this.logger.logError(error, 'UsersService.create');
      if (error instanceof ConflictException) {
        throw error;
      }
      throw new BadRequestException('Failed to create user');
    }
  }

  async findAll(options?: {
    page?: number;
    limit?: number;
    teamId?: string;
    search?: string;
  }): Promise<{ users: any[]; total: number; page: number; limit: number }> {
    const { page = 1, limit = 20, teamId, search } = options || {};
    const skip = (page - 1) * limit;

    try {
      const where: any = {};

      if (teamId) {
        where.teamId = teamId;
      }

      if (search) {
        where.OR = [
          {
            firstName: {
              contains: search,
              mode: 'insensitive',
            },
          },
          {
            lastName: {
              contains: search,
              mode: 'insensitive',
            },
          },
          {
            email: {
              contains: search,
              mode: 'insensitive',
            },
          },
        ];
      }

      const [users, total] = await Promise.all([
        this.databaseService.user.findMany({
          where,
          skip,
          take: limit,
          orderBy: [
            { level: 'desc' },
            { xp: 'desc' },
            { createdAt: 'desc' },
          ],
          select: {
            id: true,
            email: true,
            firstName: true,
            lastName: true,
            avatar: true,
            level: true,
            xp: true,
            streak: true,
            totalFocusTime: true,
            tasksCompleted: true,
            qualityScore: true,
            wellnessScore: true,
            teamId: true,
            preferences: true,
            createdAt: true,
            updatedAt: true,
          },
        }),
        this.databaseService.user.count({ where }),
      ]);

      const formattedUsers = users.map((user) => ({
        ...user,
        preferences: user.preferences ? JSON.parse(user.preferences as string) : null,
      }));

      return {
        users: formattedUsers,
        total,
        page,
        limit,
      };
    } catch (error) {
      this.logger.logError(error, 'UsersService.findAll');
      throw new BadRequestException('Failed to fetch users');
    }
  }

  async findById(id: string): Promise<any> {
    try {
      const user = await this.databaseService.user.findUnique({
        where: { id },
        select: {
          id: true,
          email: true,
          firstName: true,
          lastName: true,
          avatar: true,
          level: true,
          xp: true,
          streak: true,
          totalFocusTime: true,
          tasksCompleted: true,
          qualityScore: true,
          wellnessScore: true,
          teamId: true,
          preferences: true,
          createdAt: true,
          updatedAt: true,
        },
      });

      if (!user) {
        throw new NotFoundException(`User with ID ${id} not found`);
      }

      return {
        ...user,
        preferences: user.preferences ? JSON.parse(user.preferences as string) : null,
      };
    } catch (error) {
      if (error instanceof NotFoundException) {
        throw error;
      }
      this.logger.logError(error, 'UsersService.findById');
      throw new BadRequestException('Failed to fetch user');
    }
  }

  async findByEmail(email: string): Promise<any> {
    try {
      const user = await this.databaseService.user.findUnique({
        where: { email },
        select: {
          id: true,
          email: true,
          firstName: true,
          lastName: true,
          avatar: true,
          level: true,
          xp: true,
          streak: true,
          totalFocusTime: true,
          tasksCompleted: true,
          qualityScore: true,
          wellnessScore: true,
          teamId: true,
          preferences: true,
          createdAt: true,
          updatedAt: true,
        },
      });

      if (!user) {
        throw new NotFoundException(`User with email ${email} not found`);
      }

      return {
        ...user,
        preferences: user.preferences ? JSON.parse(user.preferences as string) : null,
      };
    } catch (error) {
      if (error instanceof NotFoundException) {
        throw error;
      }
      this.logger.logError(error, 'UsersService.findByEmail');
      throw new BadRequestException('Failed to fetch user by email');
    }
  }

  async update(id: string, updateUserDto: UpdateUserDto): Promise<any> {
    try {
      // Check if user exists
      const existingUser = await this.databaseService.user.findUnique({
        where: { id },
      });

      if (!existingUser) {
        throw new NotFoundException(`User with ID ${id} not found`);
      }

      // Check if email is being changed and if it's already taken
      if (updateUserDto.email && updateUserDto.email !== existingUser.email) {
        const emailExists = await this.databaseService.user.findUnique({
          where: { email: updateUserDto.email },
        });

        if (emailExists) {
          throw new ConflictException('Email is already in use by another user');
        }
      }

      // Prepare update data
      const updateData: any = {};

      if (updateUserDto.email) updateData.email = updateUserDto.email;
      if (updateUserDto.firstName) updateData.firstName = updateUserDto.firstName;
      if (updateUserDto.lastName) updateData.lastName = updateUserDto.lastName;
      if (updateUserDto.avatar !== undefined) updateData.avatar = updateUserDto.avatar;
      if (updateUserDto.level !== undefined) updateData.level = updateUserDto.level;
      if (updateUserDto.xp !== undefined) updateData.xp = updateUserDto.xp;
      if (updateUserDto.streak !== undefined) updateData.streak = updateUserDto.streak;
      if (updateUserDto.totalFocusTime !== undefined) updateData.totalFocusTime = updateUserDto.totalFocusTime;
      if (updateUserDto.tasksCompleted !== undefined) updateData.tasksCompleted = updateUserDto.tasksCompleted;
      if (updateUserDto.qualityScore !== undefined) updateData.qualityScore = updateUserDto.qualityScore;
      if (updateUserDto.wellnessScore !== undefined) updateData.wellnessScore = updateUserDto.wellnessScore;
      if (updateUserDto.teamId !== undefined) updateData.teamId = updateUserDto.teamId;

      if (updateUserDto.preferences !== undefined) {
        updateData.preferences = updateUserDto.preferences ? JSON.stringify(updateUserDto.preferences) : null;
      }

      const updatedUser = await this.databaseService.user.update({
        where: { id },
        data: updateData,
        select: {
          id: true,
          email: true,
          firstName: true,
          lastName: true,
          avatar: true,
          level: true,
          xp: true,
          streak: true,
          totalFocusTime: true,
          tasksCompleted: true,
          qualityScore: true,
          wellnessScore: true,
          teamId: true,
          preferences: true,
          createdAt: true,
          updatedAt: true,
        },
      });

      this.logger.logUserAction('USER_UPDATED', updatedUser.id, {
        email: updatedUser.email,
        changedFields: Object.keys(updateData),
      });

      return {
        ...updatedUser,
        preferences: JSON.parse(updatedUser.preferences as string),
      };
    } catch (error) {
      if (error instanceof NotFoundException || error instanceof ConflictException) {
        throw error;
      }
      this.logger.logError(error, 'UsersService.update');
      throw new BadRequestException('Failed to update user');
    }
  }

  async remove(id: string): Promise<void> {
    try {
      // Check if user exists
      const existingUser = await this.databaseService.user.findUnique({
        where: { id },
      });

      if (!existingUser) {
        throw new NotFoundException(`User with ID ${id} not found`);
      }

      await this.databaseService.user.delete({
        where: { id },
      });

      this.logger.logUserAction('USER_DELETED', id, {
        email: existingUser.email,
      });
    } catch (error) {
      if (error instanceof NotFoundException) {
        throw error;
      }
      this.logger.logError(error, 'UsersService.remove');
      throw new BadRequestException('Failed to delete user');
    }
  }

  async getUserStatistics(userId: string): Promise<any> {
    try {
      const user = await this.databaseService.user.findUnique({
        where: { id: userId },
        select: {
          id: true,
          level: true,
          xp: true,
          streak: true,
          totalFocusTime: true,
          tasksCompleted: true,
          qualityScore: true,
          wellnessScore: true,
          createdAt: true,
        },
      });

      if (!user) {
        throw new NotFoundException(`User with ID ${userId} not found`);
      }

      // Get additional statistics from sessions and tasks
      const [
        sessionStats,
        taskStats,
        recentSessions,
      ] = await Promise.all([
        this.databaseService.session.aggregate({
          where: { userId },
          _count: { id: true },
          _sum: { duration: true },
        }),
        this.databaseService.task.aggregate({
          where: { assigneeId: userId },
          _count: { id: true },
          _avg: { completedPomodoros: true },
        }),
        this.databaseService.session.findMany({
          where: { userId },
          orderBy: { startTime: 'desc' },
          take: 5,
          select: {
            id: true,
            type: true,
            duration: true,
            startTime: true,
            completed: true,
          },
        }),
      ]);

      const avgSessionDuration = (sessionStats._sum.duration || 0) / (sessionStats._count.id || 1);
      const avgTaskCompletion = taskStats._avg.completedPomodoros || 0;

      return {
        user: {
          id: user.id,
          level: user.level,
          xp: user.xp,
          streak: user.streak,
          memberSince: user.createdAt,
        },
        productivity: {
          totalFocusTime: user.totalFocusTime,
          tasksCompleted: user.tasksCompleted,
          qualityScore: user.qualityScore,
          wellnessScore: user.wellnessScore,
        },
        sessions: {
          totalSessions: sessionStats._count.id,
          totalSessionDuration: sessionStats._sum.duration || 0,
          averageSessionDuration: Math.round(avgSessionDuration),
          recentSessions,
        },
        tasks: {
          totalTasks: taskStats._count.id,
          averageTaskCompletion: Math.round(avgTaskCompletion * 100) / 100,
        },
      };
    } catch (error) {
      if (error instanceof NotFoundException) {
        throw error;
      }
      this.logger.logError(error, 'UsersService.getUserStatistics');
      throw new BadRequestException('Failed to fetch user statistics');
    }
  }

  async updateUserLevel(userId: string, xpGained: number): Promise<any> {
    try {
      const user = await this.databaseService.user.findUnique({
        where: { id: userId },
        select: { xp: true, level: true },
      });

      if (!user) {
        throw new NotFoundException(`User with ID ${userId} not found`);
      }

      const newXP = user.xp + xpGained;
      const newLevel = Math.floor(newXP / 100) + 1; // Simple leveling: every 100 XP = 1 level

      const updatedUser = await this.databaseService.user.update({
        where: { id: userId },
        data: {
          xp: newXP,
          level: newLevel,
        },
        select: {
          id: true,
          email: true,
          level: true,
          xp: true,
          firstName: true,
          lastName: true,
        },
      });

      this.logger.logUserAction('USER_LEVEL_UPDATED', userId, {
        oldLevel: user.level,
        newLevel: newLevel,
        xpGained,
        totalXP: newXP,
      });

      return updatedUser;
    } catch (error) {
      if (error instanceof NotFoundException) {
        throw error;
      }
      this.logger.logError(error, 'UsersService.updateUserLevel');
      throw new BadRequestException('Failed to update user level');
    }
  }

  async updatePreferences(userId: string, updatePreferencesDto: UpdatePreferencesDto, replaceAll = false) {
    try {
      // Check if user exists
      const existingUser = await this.databaseService.user.findUnique({
        where: { id: userId },
      });

      if (!existingUser) {
        throw new NotFoundException(`User with ID ${userId} not found`);
      }

      // Parse existing preferences or create new ones
      let currentPreferences: any = {};
      if (existingUser.preferences) {
        try {
          currentPreferences = JSON.parse(existingUser.preferences);
        } catch (parseError) {
          this.logger.logError(parseError, 'UsersService.updatePreferences - parsing existing preferences');
          currentPreferences = {};
        }
      }

      // For PUT semantics, replace all preferences; for PATCH semantics, merge
      const updatedPreferences = replaceAll
        ? updatePreferencesDto
        : {
            ...currentPreferences,
            ...updatePreferencesDto,
          };

      // Update user with new preferences
      const updatedUser = await this.databaseService.user.update({
        where: { id: userId },
        data: {
          preferences: JSON.stringify(updatedPreferences),
        },
      });

      this.logger.log(`Updated preferences for user ${userId}`, 'UsersService.updatePreferences');

      // Return user with parsed preferences
      return {
        ...updatedUser,
        preferences: updatedPreferences,
      };
    } catch (error) {
      if (error instanceof NotFoundException) {
        throw error;
      }
      this.logger.logError(error, 'UsersService.updatePreferences');
      throw new BadRequestException('Failed to update user preferences');
    }
  }
}