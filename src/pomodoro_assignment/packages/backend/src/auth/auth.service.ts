import { Injectable, ConflictException, UnauthorizedException, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { JwtService } from '@nestjs/jwt';
import * as bcrypt from 'bcryptjs';
import { DatabaseService } from '../config/database.config';
import { RegisterDto, LoginDto } from './dto';
import { AuthResponseDto } from './dto/auth-response.dto';
import { LoggerService } from '@/core/logger/logger.service';

@Injectable()
export class AuthService {
  constructor(
    private readonly databaseService: DatabaseService,
    private readonly jwtService: JwtService,
    private readonly configService: ConfigService,
    private readonly logger: LoggerService,
  ) {}

  async register(registerDto: RegisterDto): Promise<AuthResponseDto> {
    const { email, password, firstName, lastName, avatar, teamId } = registerDto;

    // Check if user already exists
    const existingUser = await this.databaseService.user.findUnique({
      where: { email },
    });

    if (existingUser) {
      this.logger.debug('Registration attempt with existing email');
      throw new ConflictException('Email already registered');
    }

    // Hash password with bcrypt
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    // Set default preferences
    const defaultPreferences = {
      theme: 'system',
      pomodoroLength: 25,
      shortBreakLength: 5,
      longBreakLength: 15,
      notifications: {
        desktop: true,
        sound: true,
        email: false,
      },
      autoStartBreaks: false,
      autoStartPomodoros: false,
    };

    try {
      // Create user
      const user = await this.databaseService.user.create({
        data: {
          email,
          password:hashedPassword,
          firstName,
          lastName,
          
          avatar: avatar || null,
          teamId: teamId || null,
          preferences: JSON.stringify(defaultPreferences),
        },
        select: {
          id: true,
          email: true,
          firstName: true,
          lastName: true,
          
          avatar: true,
          level: true,
          xp: true,
          streak: true,
          teamId: true,
          createdAt: true,
        },
      });

       

      // Generate tokens
      const { token, refreshToken } = await this.generateTokens(user);

      // Add user to team if teamId provided
      if (teamId) {
        try {
          await this.databaseService.teamMember.create({
            data: {
              userId: user.id,
              teamId,
              role: 'MEMBER',
            },
          });

          this.logger.log('TEAM_JOINED', user.id, {
            teamId,
          });
        } catch (error) {
          // Log warning but don't fail registration
          this.logger.warn('Failed to add user to team during registration', 'AuthService.register', {
            userId: user.id,
            teamId,
            error: (error as Error).message,
          });
        }
      }

      return {
        token,
        refreshToken,
        user: {
          id: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          avatar: user.avatar || undefined,
          level: user.level,
          xp: user.xp,
          streak: user.streak,
          teamId: user.teamId || undefined,
        },
      };
    } catch (error) {
      this.logger.error(error, 'AuthService.register');
      throw new ConflictException('Failed to create user account');
    }
  }

  async login(loginDto: LoginDto): Promise<AuthResponseDto> {
    const { email, password: rawPassword } = loginDto;

    // Find user with password hash
    const user = await this.databaseService.user.findUnique({
      where: { email },
      select: {
        id: true,
        email: true,
        password: true,
        firstName: true,
        lastName: true,
        avatar: true,
        level: true,
        xp: true,
        streak: true,
        teamId: true,
      },
    });

    if (!user) {
      this.logger.debug('Login attempt with non-existent email',);
      throw new UnauthorizedException('Invalid credentials');
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(rawPassword, user.password);
    if (!isPasswordValid) {
      this.logger.debug('Login attempt with invalid password', );
      throw new UnauthorizedException('Invalid credentials');
    }

    // Remove password from response
    const { password: _, ...userWithoutPassword } = user;

    this.logger.log('USER_LOGGED_IN', user.id, {
      email: user.email,
    });

    // Generate tokens
    const { token, refreshToken } = await this.generateTokens(userWithoutPassword);

    return {
      token,
      refreshToken,
      user: {
        id: userWithoutPassword.id,
        email: userWithoutPassword.email,
        firstName: userWithoutPassword.firstName,
        lastName: userWithoutPassword.lastName,
        avatar: userWithoutPassword.avatar || undefined,
        level: userWithoutPassword.level,
        xp: userWithoutPassword.xp,
        streak: userWithoutPassword.streak,
        teamId: userWithoutPassword.teamId || undefined,
      },
    };
  }

  async refreshToken(refreshToken: string): Promise<AuthResponseDto> {
    try {
      // Verify refresh token
      const payload = this.jwtService.verify(refreshToken);

      // Find user
      const user = await this.databaseService.user.findUnique({
        where: { id: payload.sub },
        select: {
          id: true,
          email: true,
          firstName: true,
          lastName: true,
          avatar: true,
          level: true,
          xp: true,
          streak: true,
          teamId: true,
        },
      });

      if (!user) {
        throw new UnauthorizedException('Invalid refresh token');
      }

      this.logger.log('TOKEN_REFRESHED', user.id);

      // Generate new tokens
      const { token, refreshToken: newRefreshToken } = await this.generateTokens(user);

      return {
        token,
        refreshToken: newRefreshToken,
        user: {
          id: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          avatar: user.avatar || undefined,
          level: user.level,
          xp: user.xp,
          streak: user.streak,
          teamId: user.teamId || undefined,
        },
      };
    } catch (error) {
      this.logger.debug('Invalid refresh token attempt', );
      throw new UnauthorizedException('Invalid refresh token');
    }
  }

  async validateUser(userId: string) {
    const user = await this.databaseService.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        firstName: true,
        lastName: true,
        avatar: true,
        level: true,
        xp: true,
        streak: true,
        teamId: true,
        preferences: true,
      },
    });

    if (!user) {
      return null;
    }

    return user;
  }

  async changePassword(userId: string, currentPassword: string, newPassword: string): Promise<void> {
    const user = await this.databaseService.user.findUnique({
      where: { id: userId },
      select: { password: true },
    });

    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Verify current password
    const isCurrentPasswordValid = await bcrypt.compare(currentPassword, user.password);
    if (!isCurrentPasswordValid) {
      throw new UnauthorizedException('Current password is incorrect');
    }

    // Hash new password
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(newPassword, saltRounds);

    // Update password
    await this.databaseService.user.update({
      where: { id: userId },
      data: { password: hashedPassword },
    });

    this.logger.log('PASSWORD_CHANGED', userId);
  }

  private async generateTokens(user: any): Promise<{ token: string; refreshToken: string }> {
    const payload = {
      sub: user.id,
      email: user.email,
    };

    const token = this.jwtService.sign(payload);
    const refreshToken = this.jwtService.sign(payload, { expiresIn: '7d' });

    return { token, refreshToken };
  }
}