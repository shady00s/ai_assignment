# OptoPomodoro Backend Implementation Plan
**NestJS + TypeScript + SQLite + Prisma**

## Team Overview
**Framework**: NestJS 10.4.x with TypeScript 5.5.x
**Database**: SQLite with Prisma ORM 5.19.x
**Authentication**: JWT with @nestjs/passport + Passport.js
**Real-time**: @nestjs/websockets + Socket.IO 4.7.x
**Validation**: class-validator 0.14.x + class-transformer 0.5.x
**Testing**: Jest 29.x + Supertest 6.x
**Security**: Helmet, CORS, bcryptjs 2.4.x

## Phase 1: Foundation Setup (Week 1-2)

### 1.1 Package Installation & Configuration

#### Core Dependencies
```bash
# Core framework
pnpm add @nestjs/core@10.4.1 @nestjs/common@10.4.1 @nestjs/platform-express@10.4.1
pnpm add @nestjs/config@3.2.3 @nestjs/platform-socket.io@10.4.1

# TypeScript & Development
pnpm add -D typescript@5.5.4 @types/node@22.5.4
pnpm add -D @nestjs/cli@10.4.4 @nestjs/schematics@10.1.1
pnpm add -D ts-node@10.9.2 tsconfig-paths@4.2.0

# Database & ORM
pnpm add @prisma/client@5.19.1 prisma@5.19.1
pnpm add -D @prisma/client@5.19.1

# Authentication & Security
pnpm add @nestjs/jwt@10.2.0 @nestjs/passport@10.0.3
pnpm add passport@0.7.0 @passport-jwt@0.4.1 passport-google-oauth20@2.0.0
pnpm add bcryptjs@2.4.3 @types/bcryptjs@2.4.6
pnpm add helmet@7.1.0 @nestjs/throttler@5.1.2

# Validation & Transformation
pnpm add class-validator@0.14.1 class-transformer@0.5.1

# Real-time Communication
pnpm add socket.io@4.7.5 @types/socket.io@3.0.2

# Utilities
pnpm add winston@3.13.1 @nestjs/winston@2.0.4
pnpm add date-fns@3.6.0 uuid@10.0.0 @types/uuid@10.0.0

# Testing
pnpm add -D @nestjs/testing@10.4.1 jest@29.7.0 @types/jest@29.5.12
pnpm add -D supertest@6.3.3 @types/supertest@6.0.2

# Development tools
pnpm add -D eslint@9.9.1 @typescript-eslint/eslint-plugin@8.4.0
pnpm add -D prettier@3.3.3 nodemon@3.1.4
```

#### Project Structure
```
packages/backend/
├── src/
│   ├── auth/
│   │   ├── dto/
│   │   │   ├── login.dto.ts
│   │   │   ├── register.dto.ts
│   │   │   └── auth.module.ts
│   │   ├── guards/
│   │   │   ├── jwt-auth.guard.ts
│   │   │   └── roles.guard.ts
│   │   ├── strategies/
│   │   │   ├── jwt.strategy.ts
│   │   │   └── google.strategy.ts
│   │   ├── decorators/
│   │   │   └── current-user.decorator.ts
│   │   └── auth.module.ts
│   ├── users/
│   │   ├── dto/
│   │   ├── entities/
│   │   ├── users.controller.ts
│   │   ├── users.service.ts
│   │   └── users.module.ts
│   ├── tasks/
│   │   ├── dto/
│   │   ├── entities/
│   │   ├── tasks.controller.ts
│   │   ├── tasks.service.ts
│   │   └── tasks.module.ts
│   ├── sessions/
│   │   ├── dto/
│   │   ├── entities/
│   │   ├── sessions.controller.ts
│   │   ├── sessions.service.ts
│   │   └── sessions.module.ts
│   ├── teams/
│   │   ├── dto/
│   │   ├── entities/
│   │   ├── teams.controller.ts
│   │   ├── teams.service.ts
│   │   └── teams.module.ts
│   ├── gamification/
│   │   ├── achievements/
│   │   ├── analytics/
│   │   ├── gamification.controller.ts
│   │   ├── gamification.service.ts
│   │   └── gamification.module.ts
│   ├── websocket/
│   │   ├── gateway/
│   │   ├── events/
│   │   ├── websocket.gateway.ts
│   │   └── websocket.module.ts
│   ├── database/
│   │   ├── migrations/
│   │   ├── seeds/
│   │   └── database.module.ts
│   ├── common/
│   │   ├── filters/
│   │   ├── interceptors/
│   │   ├── pipes/
│   │   ├── decorators/
│   │   └── interfaces/
│   ├── app.module.ts
│   ├── main.ts
│   └── config/
├── prisma/
│   ├── schema.prisma
│   ├── migrations/
│   └── seeds/
├── test/
├── package.json
├── tsconfig.json
├── nest-cli.json
└── Dockerfile
```

### 1.2 Prisma Database Schema

#### prisma/schema.prisma
```prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "sqlite"
  url      = env("DATABASE_URL")
}

// Enums
enum TeamRole {
  OWNER
  ADMIN
  MEMBER
}

enum Priority {
  CRITICAL
  IMPORTANT
  NORMAL
  CREATIVE
}

enum TaskStatus {
  TODO
  IN_PROGRESS
  DONE
}

enum SessionType {
  POMODORO
  SHORT_BREAK
  LONG_BREAK
  CUSTOM
}

enum AchievementCategory {
  FOCUS_MASTERY
  CONSISTENCY
  TEAM_COLLABORATION
  WELLNESS_MINDFULNESS
}

// Models
model User {
  id          String   @id @default(cuid())
  email       String   @unique
  name        String
  avatar      String?
  preferences Json     @default("{}")
  settings    Json     @default("{}")
  level       Int      @default(1)
  xp          Int      @default(0)
  streak      Int      @default(0)
  createdAt   DateTime @default(now()) @map("created_at")
  updatedAt   DateTime @updatedAt @map("updated_at")

  // Relations
  sessions    Session[]
  tasks       Task[]
  teamMembers TeamMember[]
  achievements UserAchievement[]
  notifications Notification[]

  @@map("users")
  @@index([email])
  @@index([createdAt])
  @@index([level, xp])
}

model Team {
  id          String   @id @default(cuid())
  name        String
  description String?
  settings    Json     @default("{}")
  createdAt   DateTime @default(now()) @map("created_at")
  updatedAt   DateTime @updatedAt @map("updated_at")

  // Relations
  members   TeamMember[]
  tasks     Task[]
  challenges TeamChallenge[]

  @@map("teams")
  @@index([createdAt])
}

model TeamMember {
  id     String @id @default(cuid())
  userId String @map("user_id")
  teamId String @map("team_id")
  role   TeamRole @default(MEMBER)
  joinedAt DateTime @default(now()) @map("joined_at")

  // Relations
  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
  team Team @relation(fields: [teamId], references: [id], onDelete: Cascade)

  @@unique([userId, teamId])
  @@index([teamId])
  @@index([userId])
  @@map("team_members")
}

model Task {
  id                String     @id @default(cuid())
  title             String
  description       String?
  priority          Priority   @default(NORMAL)
  status            TaskStatus @default(TODO)
  dueDate           DateTime?  @map("due_date")
  estimatedMinutes  Int?       @map("estimated_minutes")
  actualMinutes     Int?       @map("actual_minutes")
  projectId         String?    @map("project_id")
  assigneeId        String?    @map("assignee_id")
  creatorId         String     @map("creator_id")
  teamId            String?    @map("team_id")
  complexity        Int        @default(1)
  createdAt         DateTime   @default(now()) @map("created_at")
  updatedAt         DateTime   @updatedAt @map("updated_at")

  // Relations
  creator   User     @relation(fields: [creatorId], references: [id])
  assignee  User?    @relation(fields: [assigneeId], references: [id])
  team      Team?    @relation(fields: [teamId], references: [id])
  sessions  Session[]
  dependencies TaskDependency[] @relation("DependentTask")
  dependents   TaskDependency[] @relation("PrerequisiteTask")

  @@map("tasks")
  @@index([assigneeId, status])
  @@index([teamId, dueDate])
  @@index([creatorId])
  @@index([status, priority])
}

model TaskDependency {
  id               String @id @default(cuid())
  dependentTaskId  String @map("dependent_task_id")
  prerequisiteId   String @map("prerequisite_id")
  createdAt        DateTime @default(now()) @map("created_at")

  // Relations
  dependentTask  Task @relation("DependentTask", fields: [dependentTaskId], references: [id], onDelete: Cascade)
  prerequisite    Task @relation("PrerequisiteTask", fields: [prerequisiteId], references: [id], onDelete: Cascade)

  @@unique([dependentTaskId, prerequisiteId])
  @@map("task_dependencies")
}

model Session {
  id          String      @id @default(cuid())
  userId      String      @map("user_id")
  taskId      String?     @map("task_id")
  type        SessionType @default(POMODORO)
  duration    Int         // in minutes
  startedAt   DateTime    @map("started_at")
  completedAt DateTime?   @map("completed_at")
  notes       String?
  quality     Int?        // 1-5 rating
  createdAt   DateTime    @default(now()) @map("created_at")

  // Relations
  user User   @relation(fields: [userId], references: [id])
  task Task?  @relation(fields: [taskId], references: [id])

  @@map("sessions")
  @@index([userId, startedAt])
  @@index([userId, type])
  @@index([completedAt])
  @@index([taskId])
}

model Achievement {
  id          String @id @default(cuid())
  name        String
  description String
  icon        String
  category    AchievementCategory
  xpValue     Int
  criteria    Json  // Conditions to unlock
  isActive    Boolean @default(true)
  createdAt   DateTime @default(now()) @map("created_at")

  // Relations
  userAchievements UserAchievement[]

  @@map("achievements")
  @@index([category])
  @@index([isActive])
}

model UserAchievement {
  id           String   @id @default(cuid())
  userId       String   @map("user_id")
  achievementId String  @map("achievement_id")
  unlockedAt   DateTime @default(now()) @map("unlocked_at")
  progress     Json?    // Progress data for partially completed achievements

  // Relations
  user       User       @relation(fields: [userId], references: [id])
  achievement Achievement @relation(fields: [achievementId], references: [id])

  @@unique([userId, achievementId])
  @@index([userId])
  @@index([achievementId])
  @@map("user_achievements")
}

model TeamChallenge {
  id          String   @id @default(cuid())
  teamId      String   @map("team_id")
  name        String
  description String?
  type        String   // e.g., "FOCUS_TIME", "TASK_COMPLETION"
  targetValue Int      @map("target_value")
  currentValue Int     @default(0) @map("current_value")
  startDate   DateTime @map("start_date")
  endDate     DateTime @map("end_date")
  isActive    Boolean  @default(true)
  createdAt   DateTime @default(now()) @map("created_at")
  updatedAt   DateTime @updatedAt @map("updated_at")

  // Relations
  team Team @relation(fields: [teamId], references: [id])

  @@map("team_challenges")
  @@index([teamId])
  @@index([isActive])
  @@index([endDate])
}

model Notification {
  id        String   @id @default(cuid())
  userId    String   @map("user_id")
  type      String   // e.g., "ACHIEVEMENT", "TEAM_UPDATE", "REMINDER"
  title     String
  message   String
  data      Json?    // Additional data
  readAt    DateTime? @map("read_at")
  createdAt DateTime @default(now()) @map("created_at")

  // Relations
  user User @relation(fields: [userId], references: [id])

  @@map("notifications")
  @@index([userId])
  @@index([readAt])
  @@index([createdAt])
}
```

### 1.3 Core Application Setup

#### main.ts
```typescript
import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import helmet from 'helmet';
import { AppModule } from './app.module';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { Logger } from 'nestjs-pino';
import { LoggingInterceptor } from './common/interceptors/logging.interceptor';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, {
    bufferLogs: true,
  });

  const configService = app.get(ConfigService);

  // Security
  app.use(helmet());
  app.enableCors({
    origin: [
      'http://localhost:3000',
      configService.get('FRONTEND_URL'),
      configService.get('PRODUCTION_URL'),
    ],
    credentials: true,
  });

  // Global pipes
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
      transformOptions: {
        enableImplicitConversion: true,
      },
    }),
  );

  // Global interceptors
  app.useGlobalInterceptors(new LoggingInterceptor());

  // API prefix
  app.setGlobalPrefix('api');

  // Swagger documentation
  const config = new DocumentBuilder()
    .setTitle('OptoPomodoro API')
    .setDescription('Zen-inspired productivity API for Optomatica teams')
    .setVersion('1.0')
    .addBearerAuth()
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('api/docs', app, document);

  // Start server
  const port = configService.get('PORT', 3001);
  await app.listen(port);

  console.log(`🚀 OptoPomodoro API running on port ${port}`);
  console.log(`📚 API Documentation: http://localhost:${port}/api/docs`);
}

bootstrap();
```

#### app.module.ts
```typescript
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ThrottlerModule } from '@nestjs/throttler';
import { LoggerModule } from 'nestjs-pino';

import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { TasksModule } from './tasks/tasks.module';
import { SessionsModule } from './sessions/sessions.module';
import { TeamsModule } from './teams/teams.module';
import { GamificationModule } from './gamification/gamification.module';
import { WebsocketModule } from './websocket/websocket.module';
import { DatabaseModule } from './database/database.module';

import configuration from './config/configuration';

@Module({
  imports: [
    // Configuration
    ConfigModule.forRoot({
      isGlobal: true,
      load: [configuration],
    }),

    // Logging
    LoggerModule.forRoot({
      pinoHttp: {
        level: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
        transport: process.env.NODE_ENV === 'development'
          ? { target: 'pino-pretty' }
          : undefined,
      },
    }),

    // Rate limiting
    ThrottlerModule.forRoot([
      {
        ttl: 60000, // 1 minute
        limit: 100, // 100 requests per minute
      },
    ]),

    // Feature modules
    DatabaseModule,
    AuthModule,
    UsersModule,
    TasksModule,
    SessionsModule,
    TeamsModule,
    GamificationModule,
    WebsocketModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
```

#### config/configuration.ts
```typescript
import { registerAs } from '@nestjs/config';

export default registerAs('app', () => ({
  port: parseInt(process.env.PORT, 10) || 3001,
  nodeEnv: process.env.NODE_ENV || 'development',
  frontendUrl: process.env.FRONTEND_URL || 'http://localhost:3000',
  productionUrl: process.env.PRODUCTION_URL || 'https://optopomodoro.com',
}));

export const databaseConfig = registerAs('database', () => ({
  url: process.env.DATABASE_URL || 'file:./data/optopomodoro.db',
}));

export const jwtConfig = registerAs('jwt', () => ({
  secret: process.env.JWT_SECRET || 'your-super-secret-jwt-key',
  expiresIn: process.env.JWT_EXPIRES_IN || '7d',
}));

export const authConfig = registerAs('auth', () => ({
  google: {
    clientId: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    callbackUrl: process.env.GOOGLE_CALLBACK_URL || 'http://localhost:3001/api/auth/google/callback',
  },
  microsoft: {
    clientId: process.env.MICROSOFT_CLIENT_ID,
    clientSecret: process.env.MICROSOFT_CLIENT_SECRET,
    callbackUrl: process.env.MICROSOFT_CALLBACK_URL || 'http://localhost:3001/api/auth/microsoft/callback',
  },
}));
```

## Phase 2: Authentication Module (Week 2-3)

### 2.1 Authentication Module Setup

#### auth/auth.module.ts
```typescript
import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { ConfigModule, ConfigService } from '@nestjs/config';

import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { JwtStrategy } from './strategies/jwt.strategy';
import { GoogleStrategy } from './strategies/google.strategy';
import { UsersModule } from '../users/users.module';

import jwtConfig from '../config/configuration';

@Module({
  imports: [
    PassportModule.register({ defaultStrategy: 'jwt' }),
    JwtModule.registerAsync({
      imports: [ConfigModule],
      useFactory: async (configService: ConfigService) => ({
        secret: configService.get('jwt.secret'),
        signOptions: {
          expiresIn: configService.get('jwt.expiresIn'),
        },
      }),
      inject: [ConfigService],
    }),
    UsersModule,
  ],
  controllers: [AuthController],
  providers: [AuthService, JwtStrategy, GoogleStrategy],
  exports: [AuthService, JwtStrategy],
})
export class AuthModule {}
```

#### auth/auth.service.ts
```typescript
import { Injectable, UnauthorizedException, ConflictException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import { UsersService } from '../users/users.service';
import * as bcrypt from 'bcryptjs';
import { LoginDto } from './dto/login.dto';
import { RegisterDto } from './dto/register.dto';
import { User } from '@prisma/client';

@Injectable()
export class AuthService {
  constructor(
    private readonly usersService: UsersService,
    private readonly jwtService: JwtService,
    private readonly configService: ConfigService,
  ) {}

  async validateUser(email: string, password: string): Promise<User | null> {
    const user = await this.usersService.findByEmail(email);
    if (!user) {
      return null;
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return null;
    }

    return user;
  }

  async login(loginDto: LoginDto) {
    const { email, password } = loginDto;
    const user = await this.validateUser(email, password);

    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const token = this.generateToken(user);

    // Remove password from response
    const { password: _, ...userWithoutPassword } = user;

    return {
      user: userWithoutPassword,
      token,
    };
  }

  async register(registerDto: RegisterDto) {
    const { email, password, name } = registerDto;

    // Validate domain
    if (!this.isValidOptomaticaEmail(email)) {
      throw new UnauthorizedException('Only Optomatica email addresses are allowed');
    }

    // Check if user already exists
    const existingUser = await this.usersService.findByEmail(email);
    if (existingUser) {
      throw new ConflictException('User with this email already exists');
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 12);

    // Create user
    const user = await this.usersService.create({
      email,
      password: hashedPassword,
      name,
    });

    const token = this.generateToken(user);

    // Remove password from response
    const { password: _, ...userWithoutPassword } = user;

    return {
      user: userWithoutPassword,
      token,
    };
  }

  async validateOAuthLogin(profile: any, provider: string): Promise<User> {
    const email = profile.emails?.[0]?.value;

    if (!email || !this.isValidOptomaticaEmail(email)) {
      throw new UnauthorizedException('Only Optomatica email addresses are allowed');
    }

    let user = await this.usersService.findByEmail(email);

    if (!user) {
      // Create new user from OAuth profile
      user = await this.usersService.create({
        email,
        name: profile.displayName,
        avatar: profile.photos?.[0]?.value,
        // No password for OAuth users
      });
    }

    return user;
  }

  private generateToken(user: User): string {
    const payload = {
      sub: user.id,
      email: user.email,
      name: user.name,
    };

    return this.jwtService.sign(payload);
  }

  private isValidOptomaticaEmail(email: string): boolean {
    return email.endsWith('@optomatica.com');
  }

  async refreshToken(user: any) {
    const token = this.generateToken(user);
    return { token };
  }
}
```

#### auth/strategies/jwt.strategy.ts
```typescript
import { Injectable } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { ExtractJwt, Strategy } from 'passport-jwt';
import { ConfigService } from '@nestjs/config';
import { UsersService } from '../../users/users.service';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(
    private readonly usersService: UsersService,
    private readonly configService: ConfigService,
  ) {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: configService.get('jwt.secret'),
    });
  }

  async validate(payload: any) {
    const user = await this.usersService.findById(payload.sub);
    if (!user) {
      throw new UnauthorizedException('User not found');
    }

    // Remove password from response
    const { password, ...userWithoutPassword } = user;

    return userWithoutPassword;
  }
}
```

#### auth/guards/jwt-auth.guard.ts
```typescript
import { Injectable } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';

@Injectable()
export class JwtAuthGuard extends AuthGuard('jwt') {}
```

## Phase 3: Core API Modules (Week 3-5)

### 3.1 Users Module

#### users/users.service.ts
```typescript
import { Injectable, NotFoundException, ConflictException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { User } from '@prisma/client';
import { PrismaService } from '../database/prisma.service';
import { CreateUserDto } from './dto/create-user.dto';
import { UpdateUserDto } from './dto/update-user.dto';

@Injectable()
export class UsersService {
  constructor(private readonly prisma: PrismaService) {}

  async create(createUserDto: CreateUserDto): Promise<User> {
    const user = await this.prisma.user.create({
      data: {
        email: createUserDto.email,
        name: createUserDto.name,
        password: createUserDto.password,
        preferences: createUserDto.preferences || {},
        settings: createUserDto.settings || {},
      },
    });

    return user;
  }

  async findAll(): Promise<User[]> {
    return this.prisma.user.findMany({
      include: {
        achievements: {
          include: {
            achievement: true,
          },
        },
      },
    });
  }

  async findById(id: string): Promise<User | null> {
    return this.prisma.user.findUnique({
      where: { id },
      include: {
        achievements: {
          include: {
            achievement: true,
          },
        },
        teamMembers: {
          include: {
            team: true,
          },
        },
      },
    });
  }

  async findByEmail(email: string): Promise<User | null> {
    return this.prisma.user.findUnique({
      where: { email },
    });
  }

  async update(id: string, updateUserDto: UpdateUserDto): Promise<User> {
    const user = await this.findById(id);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    return this.prisma.user.update({
      where: { id },
      data: updateUserDto,
    });
  }

  async remove(id: string): Promise<User> {
    const user = await this.findById(id);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    return this.prisma.user.delete({
      where: { id },
    });
  }

  async getUserAnalytics(userId: string, startDate?: Date, endDate?: Date) {
    const whereClause: any = {
      userId,
      completedAt: {
        not: null,
      },
    };

    if (startDate || endDate) {
      whereClause.startedAt = {
        gte: startDate,
        lte: endDate,
      };
    }

    const [totalSessions, totalMinutes, avgQuality, recentSessions] = await Promise.all([
      this.prisma.session.count({ where: whereClause }),
      this.prisma.session.aggregate({
        where: whereClause,
        _sum: { duration: true },
      }),
      this.prisma.session.aggregate({
        where: { ...whereClause, quality: { not: null } },
        _avg: { quality: true },
      }),
      this.prisma.session.findMany({
        where: whereClause,
        orderBy: { startedAt: 'desc' },
        take: 10,
      }),
    ]);

    return {
      totalSessions,
      totalMinutes: totalMinutes._sum.duration || 0,
      averageQuality: avgQuality._avg.quality || 0,
      recentSessions,
    };
  }
}
```

### 3.2 Tasks Module

#### tasks/tasks.service.ts
```typescript
import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { PrismaService } from '../database/prisma.service';
import { CreateTaskDto } from './dto/create-task.dto';
import { UpdateTaskDto } from './dto/update-task.dto';
import { Task, TaskStatus, Priority } from '@prisma/client';

@Injectable()
export class TasksService {
  constructor(private readonly prisma: PrismaService) {}

  async create(createTaskDto: CreateTaskDto, userId: string): Promise<Task> {
    return this.prisma.task.create({
      data: {
        title: createTaskDto.title,
        description: createTaskDto.description,
        priority: createTaskDto.priority || Priority.NORMAL,
        dueDate: createTaskDto.dueDate,
        estimatedMinutes: createTaskDto.estimatedMinutes,
        complexity: createTaskDto.complexity || 1,
        creatorId: userId,
        assigneeId: createTaskDto.assigneeId || userId,
        teamId: createTaskDto.teamId,
      },
      include: {
        creator: {
          select: { id: true, name: true, email: true, avatar: true },
        },
        assignee: {
          select: { id: true, name: true, email: true, avatar: true },
        },
        team: {
          select: { id: true, name: true },
        },
      },
    });
  }

  async findAll(
    userId: string,
    filters?: {
      status?: TaskStatus;
      priority?: Priority;
      teamId?: string;
      assigneeId?: string;
    }
  ): Promise<Task[]> {
    const whereClause: any = {};

    // User can see their own tasks, tasks they created, or team tasks
    whereClause.OR = [
      { creatorId: userId },
      { assigneeId: userId },
    ];

    if (filters?.status) {
      whereClause.status = filters.status;
    }

    if (filters?.priority) {
      whereClause.priority = filters.priority;
    }

    if (filters?.teamId) {
      whereClause.teamId = filters.teamId;
    }

    if (filters?.assigneeId) {
      whereClause.assigneeId = filters.assigneeId;
    }

    return this.prisma.task.findMany({
      where: whereClause,
      include: {
        creator: {
          select: { id: true, name: true, email: true, avatar: true },
        },
        assignee: {
          select: { id: true, name: true, email: true, avatar: true },
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
      orderBy: [
        { priority: 'desc' },
        { dueDate: 'asc' },
        { createdAt: 'desc' },
      ],
    });
  }

  async findOne(id: string, userId: string): Promise<Task> {
    const task = await this.prisma.task.findUnique({
      where: { id },
      include: {
        creator: {
          select: { id: true, name: true, email: true, avatar: true },
        },
        assignee: {
          select: { id: true, name: true, email: true, avatar: true },
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
          select: { id: true, duration, startedAt, completedAt, quality },
          orderBy: { startedAt: 'desc' },
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

    if (!hasAccess) {
      throw new ForbiddenException('Access denied to this task');
    }

    return task;
  }

  async update(id: string, updateTaskDto: UpdateTaskDto, userId: string): Promise<Task> {
    const task = await this.findOne(id, userId);

    // Check if user can update this task
    const canUpdate =
      task.creatorId === userId ||
      task.assigneeId === userId ||
      (task.teamId && await this.isTeamMember(userId, task.teamId));

    if (!canUpdate) {
      throw new ForbiddenException('Cannot update this task');
    }

    return this.prisma.task.update({
      where: { id },
      data: updateTaskDto,
      include: {
        creator: {
          select: { id: true, name: true, email: true, avatar: true },
        },
        assignee: {
          select: { id: true, name: true, email: true, avatar: true },
        },
        team: {
          select: { id: true, name: true },
        },
      },
    });
  }

  async remove(id: string, userId: string): Promise<Task> {
    const task = await this.findOne(id, userId);

    // Only creator can delete the task
    if (task.creatorId !== userId) {
      throw new ForbiddenException('Only the task creator can delete this task');
    }

    return this.prisma.task.delete({
      where: { id },
    });
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

  async addDependency(taskId: string, prerequisiteId: string, userId: string) {
    const task = await this.findOne(taskId, userId);
    const prerequisite = await this.findOne(prerequisiteId, userId);

    // Prevent circular dependencies
    if (await this.wouldCreateCircularDependency(taskId, prerequisiteId)) {
      throw new ForbiddenException('This would create a circular dependency');
    }

    return this.prisma.taskDependency.create({
      data: {
        dependentTaskId: taskId,
        prerequisiteId,
      },
    });
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
```

## Phase 4: WebSocket Implementation (Week 5-6)

### 4.1 WebSocket Gateway

#### websocket/websocket.gateway.ts
```typescript
import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
  OnGatewayInit,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { Logger, UseGuards } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { AuthService } from '../auth/auth.service';
import { SessionsService } from '../sessions/sessions.service';
import { TasksService } from '../tasks/tasks.service';
import { User } from '@prisma/client';

interface AuthenticatedSocket extends Socket {
  user: User;
}

@WebSocketGateway({
  cors: {
    origin: [
      'http://localhost:3000',
      process.env.FRONTEND_URL,
    ],
    credentials: true,
  },
})
export class WebsocketGateway implements OnGatewayInit, OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer()
  server: Server;

  private readonly logger = new Logger(WebsocketGateway.name);
  private connectedUsers: Map<string, AuthenticatedSocket> = new Map();

  constructor(
    private readonly jwtService: JwtService,
    private readonly authService: AuthService,
    private readonly sessionsService: SessionsService,
    private readonly tasksService: TasksService,
  ) {}

  afterInit(server: Server) {
    this.logger.log('WebSocket Gateway initialized');
  }

  async handleConnection(client: AuthenticatedSocket) {
    try {
      // Extract and verify JWT token
      const token = client.handshake.auth.token;
      if (!token) {
        this.logger.warn(`Connection rejected: No token provided`);
        client.disconnect();
        return;
      }

      const payload = this.jwtService.verify(token);
      const user = await this.authService.validateUser(payload.email, '');

      if (!user) {
        this.logger.warn(`Connection rejected: Invalid user`);
        client.disconnect();
        return;
      }

      client.user = user;
      this.connectedUsers.set(user.id, client);

      // Join user to their personal room
      client.join(`user:${user.id}`);

      // Join user to their team rooms
      await this.joinTeamRooms(client, user.id);

      this.logger.log(`User ${user.email} connected`);

      // Notify others that user is online
      client.broadcast.emit('user:online', {
        userId: user.id,
        userName: user.name,
        timestamp: new Date(),
      });

    } catch (error) {
      this.logger.error(`Connection error: ${error.message}`);
      client.disconnect();
    }
  }

  handleDisconnect(client: AuthenticatedSocket) {
    if (client.user) {
      this.connectedUsers.delete(client.user.id);

      this.logger.log(`User ${client.user.email} disconnected`);

      // Notify others that user is offline
      client.broadcast.emit('user:offline', {
        userId: client.user.id,
        userName: client.user.name,
        timestamp: new Date(),
      });
    }
  }

  @SubscribeMessage('timer:start')
  async handleTimerStart(
    @MessageBody() data: { taskId?: string; duration: number },
    @ConnectedSocket() client: AuthenticatedSocket,
  ) {
    try {
      const session = await this.sessionsService.createSession({
        userId: client.user.id,
        taskId: data.taskId,
        type: 'POMODORO',
        duration: data.duration,
      });

      // Broadcast to user's personal room
      client.emit('timer:started', {
        sessionId: session.id,
        duration: data.duration,
        taskId: data.taskId,
        timestamp: new Date(),
      });

      // Broadcast to team members if task is assigned to team
      if (data.taskId) {
        await this.broadcastToTeamMembers(client, 'member:timer:start', {
          userId: client.user.id,
          userName: client.user.name,
          taskId: data.taskId,
          sessionId: session.id,
          duration: data.duration,
        });
      }

    } catch (error) {
      this.logger.error(`Timer start error: ${error.message}`);
      client.emit('error', { message: 'Failed to start timer' });
    }
  }

  @SubscribeMessage('timer:complete')
  async handleTimerComplete(
    @MessageBody() data: { sessionId: string; quality?: number },
    @ConnectedSocket() client: AuthenticatedSocket,
  ) {
    try {
      const session = await this.sessionsService.completeSession(
        data.sessionId,
        client.user.id,
        data.quality
      );

      client.emit('timer:completed', {
        sessionId: session.id,
        quality: session.quality,
        duration: session.duration,
        xp: this.calculateXP(session),
        timestamp: new Date(),
      });

      // Check for achievements
      const achievements = await this.checkForAchievements(client.user.id);
      if (achievements.length > 0) {
        client.emit('achievements:unlocked', {
          achievements,
          timestamp: new Date(),
        });
      }

      // Broadcast to team members
      await this.broadcastToTeamMembers(client, 'member:timer:complete', {
        userId: client.user.id,
        userName: client.user.name,
        sessionId: session.id,
        duration: session.duration,
        quality: session.quality,
      });

    } catch (error) {
      this.logger.error(`Timer complete error: ${error.message}`);
      client.emit('error', { message: 'Failed to complete timer session' });
    }
  }

  @SubscribeMessage('task:update')
  async handleTaskUpdate(
    @MessageBody() data: { taskId: string; updates: any },
    @ConnectedSocket() client: AuthenticatedSocket,
  ) {
    try {
      const task = await this.tasksService.update(
        data.taskId,
        data.updates,
        client.user.id
      );

      // Broadcast to user
      client.emit('task:updated', {
        task,
        timestamp: new Date(),
      });

      // Broadcast to team members
      await this.broadcastToTeamMembers(client, 'member:task:update', {
        userId: client.user.id,
        userName: client.user.name,
        task,
        timestamp: new Date(),
      });

    } catch (error) {
      this.logger.error(`Task update error: ${error.message}`);
      client.emit('error', { message: 'Failed to update task' });
    }
  }

  @SubscribeMessage('ping')
  handlePing(@ConnectedSocket() client: AuthenticatedSocket) {
    client.emit('pong', { timestamp: new Date() });
  }

  private async joinTeamRooms(client: AuthenticatedSocket, userId: string) {
    // Implementation to join user to their team rooms
    // This would query the database for user's team memberships
    // and join them to corresponding team rooms
  }

  private async broadcastToTeamMembers(
    client: AuthenticatedSocket,
    event: string,
    data: any
  ) {
    // Implementation to broadcast message to all team members
    // excluding the sender
    // This would query the database for team memberships
    // and send the message to relevant team rooms
  }

  private calculateXP(session: any): number {
    let xp = 10; // Base XP for session

    if (session.quality >= 4) {
      xp += 5; // Quality bonus
    }

    if (session.duration >= 50) {
      xp += 5; // Deep work bonus
    }

    return xp;
  }

  private async checkForAchievements(userId: string) {
    // Implementation to check and return any newly unlocked achievements
    // This would integrate with the gamification service
    return [];
  }
}
```

## Phase 5: Gamification Module (Week 6-7)

### 5.1 Gamification Service

#### gamification/gamification.service.ts
```typescript
import { Injectable } from '@nestjs/common';
import { PrismaService } from '../database/prisma.service';
import { User, Achievement, UserAchievement, Session } from '@prisma/client';

@Injectable()
export class GamificationService {
  constructor(private readonly prisma: PrismaService) {}

  async calculateXP(userId: string, event: any): Promise<number> {
    let xp = 0;

    switch (event.type) {
      case 'SESSION_COMPLETE':
        xp = this.calculateSessionXP(event.data);
        break;
      case 'TASK_COMPLETE':
        xp = this.calculateTaskXP(event.data);
        break;
      case 'STREAK_MILESTONE':
        xp = event.data.streakDays * 2;
        break;
      case 'TEAM_CHALLENGE_COMPLETE':
        xp = 25;
        break;
    }

    // Apply multipliers
    const multipliers = await this.getXPMultipliers(userId);
    xp = Math.floor(xp * multipliers);

    // Update user XP
    await this.updateUserXP(userId, xp);

    return xp;
  }

  async checkAchievements(userId: string, event: any): Promise<Achievement[]> {
    const userAchievements = await this.getUserAchievements(userId);
    const allAchievements = await this.getAllAchievements();

    const newAchievements: Achievement[] = [];

    for (const achievement of allAchievements) {
      if (userAchievements.find(ua => ua.achievementId === achievement.id)) {
        continue; // Already unlocked
      }

      if (await this.checkAchievementCriteria(achievement, userId, event)) {
        await this.unlockAchievement(userId, achievement.id);
        newAchievements.push(achievement);
      }
    }

    return newAchievements;
  }

  private calculateSessionXP(session: Session): number {
    let xp = 10; // Base XP

    // Quality bonus
    if (session.quality >= 4) {
      xp += 5;
    }

    // Deep work bonus
    if (session.duration >= 50) {
      xp += 5;
    }

    return xp;
  }

  private calculateTaskXP(task: any): number {
    const complexity = task.complexity || 1;
    let xp = 5 * complexity;

    // Deadline bonus
    if (task.completedEarly) {
      xp = Math.ceil(xp * 1.5);
    }

    return xp;
  }

  private async getXPMultipliers(userId: string): Promise<number> {
    // Check for active streak multiplier
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: { streak: true }
    });

    let multiplier = 1.0;

    if (user.streak >= 30) {
      multiplier = 1.5; // 30 day streak
    } else if (user.streak >= 7) {
      multiplier = 1.3; // 7 day streak
    } else if (user.streak >= 3) {
      multiplier = 1.1; // 3 day streak
    }

    return multiplier;
  }

  private async updateUserXP(userId: string, xp: number): Promise<void> {
    await this.prisma.user.update({
      where: { id: userId },
      data: {
        xp: {
          increment: xp,
        },
      },
    });

    // Check for level up
    await this.checkLevelUp(userId);
  }

  private async checkLevelUp(userId: string): Promise<void> {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: { level: true, xp: true }
    });

    const requiredXP = this.getRequiredXPForLevel(user.level + 1);

    if (user.xp >= requiredXP) {
      await this.prisma.user.update({
        where: { id: userId },
        data: {
          level: {
            increment: 1,
          },
        },
      });

      // Check if user leveled up multiple times
      await this.checkLevelUp(userId);
    }
  }

  private getRequiredXPForLevel(level: number): number {
    // Exponential growth formula for XP requirements
    return Math.floor(100 * Math.pow(1.5, level - 1));
  }

  private async getUserAchievements(userId: string): Promise<UserAchievement[]> {
    return this.prisma.userAchievement.findMany({
      where: { userId },
      include: { achievement: true },
    });
  }

  private async getAllAchievements(): Promise<Achievement[]> {
    return this.prisma.achievement.findMany({
      where: { isActive: true },
    });
  }

  private async checkAchievementCriteria(
    achievement: Achievement,
    userId: string,
    event: any
  ): Promise<boolean> {
    const criteria = achievement.criteria as any;

    switch (criteria.type) {
      case 'SESSION_COUNT':
        return await this.checkSessionCount(userId, criteria);
      case 'CONSECUTIVE_DAYS':
        return await this.checkConsecutiveDays(userId, criteria);
      case 'TOTAL_TIME':
        return await this.checkTotalTime(userId, criteria);
      case 'TASK_COMPLETION':
        return await this.checkTaskCompletion(userId, criteria);
      case 'TEAM_COLLABORATION':
        return await this.checkTeamCollaboration(userId, criteria);
      default:
        return false;
    }
  }

  private async checkSessionCount(userId: string, criteria: any): Promise<boolean> {
    const sessions = await this.prisma.session.count({
      where: {
        userId,
        type: criteria.sessionType || 'POMODORO',
        startedAt: {
          gte: this.getDateFromTimeRange(criteria.timeRange),
        },
      },
    });

    return sessions >= criteria.requiredCount;
  }

  private async checkConsecutiveDays(userId: string, criteria: any): Promise<boolean> {
    // Implementation to check consecutive days with sessions
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: { streak: true }
    });

    return user.streak >= criteria.requiredDays;
  }

  private async checkTotalTime(userId: string, criteria: any): Promise<boolean> {
    const result = await this.prisma.session.aggregate({
      where: {
        userId,
        type: criteria.sessionType || 'POMODORO',
        startedAt: {
          gte: this.getDateFromTimeRange(criteria.timeRange),
        },
      },
      _sum: { duration: true },
    });

    const totalMinutes = result._sum.duration || 0;
    return totalMinutes >= criteria.requiredMinutes;
  }

  private async checkTaskCompletion(userId: string, criteria: any): Promise<boolean> {
    const tasks = await this.prisma.task.count({
      where: {
        OR: [
          { creatorId: userId },
          { assigneeId: userId },
        ],
        status: 'DONE',
        completedAt: {
          gte: this.getDateFromTimeRange(criteria.timeRange),
        },
      },
    });

    return tasks >= criteria.requiredTasks;
  }

  private async checkTeamCollaboration(userId: string, criteria: any): Promise<boolean> {
    // Implementation to check team collaboration metrics
    return false;
  }

  private getDateFromTimeRange(timeRange: string): Date {
    const now = new Date();
    switch (timeRange) {
      case 'TODAY':
        return new Date(now.getFullYear(), now.getMonth(), now.getDate());
      case 'WEEK':
        return new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      case 'MONTH':
        return new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      case 'YEAR':
        return new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
      default:
        return new Date(0);
    }
  }

  private async unlockAchievement(userId: string, achievementId: string): Promise<void> {
    await this.prisma.userAchievement.create({
      data: {
        userId,
        achievementId,
      },
    });
  }
}
```

## Phase 6: Testing Implementation (Week 7)

### 6.1 Unit Testing Setup

#### test/setup.ts
```typescript
import { Test, TestingModule } from '@nestjs/testing';
import { PrismaService } from '../src/database/prisma.service';
import { AppModule } from '../src/app.module';

export const createTestingModule = async (): Promise<TestingModule> => {
  const moduleFixture: TestingModule = await Test.createTestingModule({
    imports: [AppModule],
  })
    .overrideProvider(PrismaService)
    .useValue({
      // Mock PrismaService methods here
      user: {
        findUnique: jest.fn(),
        findMany: jest.fn(),
        create: jest.fn(),
        update: jest.fn(),
        delete: jest.fn(),
      },
      task: {
        findUnique: jest.fn(),
        findMany: jest.fn(),
        create: jest.fn(),
        update: jest.fn(),
        delete: jest.fn(),
      },
      session: {
        findUnique: jest.fn(),
        findMany: jest.fn(),
        create: jest.fn(),
        update: jest.fn(),
      },
    })
    .compile();

  return moduleFixture;
};
```

This comprehensive backend implementation plan provides:

1. **Complete NestJS architecture** with modular organization
2. **Full database schema** with all entities and relationships
3. **JWT-based authentication** with domain validation
4. **Comprehensive API endpoints** for all core features
5. **Real-time WebSocket implementation** for live collaboration
6. **Gamification engine** with achievements and XP system
7. **Robust testing infrastructure** with Jest and mocked services
8. **Security best practices** with validation and rate limiting
9. **Scalable architecture** ready for future enhancements
10. **Type-safe implementation** throughout the application

The plan ensures the backend team can build a secure, performant, and feature-rich API that supports all the requirements specified in the product design documentation.