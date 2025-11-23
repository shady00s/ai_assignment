# OptoPomodoro Technical Specification

## Executive Summary

**Technology Stack**: ReactJS + TypeScript (Frontend), ExpressJS + TypeScript (Backend), SQLite (Database)
**Architecture**: Monorepo with Progressive Web App (PWA) capabilities
**Target**: Optomatica employees with focus on team collaboration and individual productivity

This document provides comprehensive technical specifications for implementing OptoPomodoro, a Zen-inspired productivity application designed to help Optomatica employees overcome distractions and meet deadlines through mindful focus techniques, gamification, and community support.

## 1. Project Overview & Architecture

### 1.1 Technology Stack Rationale

**Frontend - ReactJS + TypeScript**:
- Component-based architecture enables modular, maintainable UI development
- TypeScript provides type safety and better developer experience
- Rich ecosystem with excellent PWA support through service workers
- Strong state management solutions (Redux Toolkit, Zustand)
- Extensive library ecosystem for animations, charts, and UI components

**Backend - ExpressJS + TypeScript**:
- Lightweight, flexible Node.js framework
- TypeScript support for type-safe API development
- Excellent WebSocket support for real-time features
- Extensive middleware ecosystem for authentication, validation, and security
- Proven scalability for team collaboration applications

**Database - SQLite**:
- Zero configuration, serverless deployment ideal for internal tool
- ACID compliance ensures data integrity
- Excellent performance for read-heavy productivity applications
- Easy migration path to PostgreSQL for future scaling
- Built-in full-text search capabilities for task search

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   React App     │  │   PWA Service   │  │   Cache Layer   │ │
│  │   (TypeScript)  │  │    Worker       │  │   (IndexedDB)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTPS/WebSocket
                                │
┌─────────────────────────────────────────────────────────────────┐
│                       API Gateway                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   ExpressJS     │  │   WebSocket     │  │   Authentication│ │
│  │   REST API      │  │   Real-time     │  │   Middleware    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Database Access
                                │
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │     SQLite      │  │   File Storage  │  │   External APIs  │ │
│  │   Database      │  │   (Audio Files) │  │   (Integrations) │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 System Design Principles

1. **Mobile-First Progressive Enhancement**: Core functionality works on all devices, enhanced features on capable devices
2. **Offline-First Architecture**: Essential timer and task functionality available without internet
3. **Real-Time Collaboration**: WebSocket-based live updates for team features
4. **Security by Design**: Zero-trust architecture with end-to-end encryption for sensitive data
5. **Performance-Driven**: Sub-second load times and 60fps animations
6. **Accessibility-First**: WCAG 2.1 AA compliance throughout the application

## 2. Monorepo Structure

### 2.1 Workspace Organization

```
optopomodoro/
├── packages/
│   ├── frontend/                 # ReactJS PWA Application
│   │   ├── public/
│   │   ├── src/
│   │   │   ├── components/       # Reusable UI components
│   │   │   ├── pages/           # Page-level components
│   │   │   ├── hooks/           # Custom React hooks
│   │   │   ├── store/           # Redux store configuration
│   │   │   ├── services/        # API and external service integrations
│   │   │   ├── utils/           # Utility functions
│   │   │   ├── types/           # TypeScript type definitions
│   │   │   └── assets/          # Static assets (images, sounds)
│   │   ├── package.json
│   │   ├── vite.config.ts       # Vite configuration
│   │   └── tsconfig.json
│   │
│   ├── backend/                  # ExpressJS REST API
│   │   ├── src/
│   │   │   ├── controllers/     # Route handlers
│   │   │   ├── middleware/      # Express middleware
│   │   │   ├── models/          # Database models
│   │   │   ├── services/        # Business logic services
│   │   │   ├── routes/          # API route definitions
│   │   │   ├── utils/           # Utility functions
│   │   │   ├── types/           # TypeScript types
│   │   │   ├── websocket/       # WebSocket handlers
│   │   │   └── config/          # Configuration files
│   │   ├── migrations/          # Database migrations
│   │   ├── seeds/              # Database seeds
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   ├── shared/                   # Shared code between frontend/backend
│   │   ├── types/               # Shared TypeScript types
│   │   ├── validation/          # Joi/Yup validation schemas
│   │   ├── constants/           # Shared constants
│   │   ├── utils/               # Shared utility functions
│   │   └── package.json
│   │
│   └── database/                 # Database-specific code
│       ├── schema/              # Database schema definitions
│       ├── migrations/          # Migration scripts
│       ├── seeds/              # Seed data
│       └── package.json
│
├── docs/                        # Documentation
├── scripts/                     # Build and deployment scripts
├── docker/                      # Docker configurations
├── package.json                 # Root package.json (workspace configuration)
├── pnpm-workspace.yaml         # PNPM workspace configuration
├── turbo.json                   # Turbo (build system) configuration
└── README.md
```

### 2.2 Package Manager Configuration

**pnpm-workspace.yaml**:
```yaml
packages:
  - 'packages/*'
```

**Root package.json**:
```json
{
  "name": "optopomodoro",
  "private": true,
  "workspaces": [
    "packages/*"
  ],
  "scripts": {
    "dev": "turbo run dev",
    "build": "turbo run build",
    "test": "turbo run test",
    "lint": "turbo run lint",
    "type-check": "turbo run type-check",
    "db:migrate": "cd packages/database && pnpm migrate",
    "db:seed": "cd packages/database && pnpm seed"
  },
  "devDependencies": {
    "turbo": "^1.10.0",
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "eslint": "^8.45.0",
    "prettier": "^3.0.0",
    "typescript": "^5.1.0"
  }
}
```

### 2.3 Build System Configuration (Turbo)

**turbo.json**:
```json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "!.next/cache/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "test": {
      "dependsOn": ["build"],
      "outputs": ["coverage/**"]
    },
    "lint": {
      "outputs": []
    },
    "type-check": {
      "dependsOn": ["^build"],
      "outputs": []
    }
  }
}
```

## 3. Frontend Architecture (ReactJS)

### 3.1 Technology Stack

**Core Framework**: React 18+ with TypeScript
**Build Tool**: Vite for fast development and optimized builds
**State Management**: Redux Toolkit with RTK Query for API state
**Routing**: React Router v6 with lazy loading
**Styling**: Styled-components with CSS-in-JS
**PWA**: Vite PWA plugin for service worker generation
**Testing**: Vitest for unit tests, Cypress for E2E tests
**Animations**: Framer Motion for smooth UI transitions

### 3.2 Application Structure

```typescript
// src/App.tsx
import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { store } from './store';
import { AppRouter } from './router';
import { ZenTheme } from './themes';
import { PWAProvider } from './providers/PWAProvider';
import { WebSocketProvider } from './providers/WebSocketProvider';

const queryClient = new QueryClient();

export const App: React.FC = () => {
  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={ZenTheme}>
          <PWAProvider>
            <WebSocketProvider>
              <BrowserRouter>
                <AppRouter />
              </BrowserRouter>
            </WebSocketProvider>
          </PWAProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </Provider>
  );
};
```

### 3.3 Component Architecture

**Atomic Design Pattern**:
```typescript
// components/atoms/
├── Button/
├── Input/
├── Icon/
├── Typography/
└── CircularProgress/

// components/molecules/
├── TaskCard/
├── TimerControls/
├── UserAvatar/
└── NotificationToast/

// components/organisms/
├── TaskBoard/
├── TimerDisplay/
├── Dashboard/
└── TeamLeaderboard/

// components/templates/
├── AppLayout/
├── AuthLayout/
└── DashboardLayout/

// components/pages/
├── HomePage/
├── TimerPage/
├── TasksPage/
├── DashboardPage/
└── TeamPage/
```

### 3.4 State Management Architecture

**Redux Store Structure**:
```typescript
// store/index.ts
import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';
import { authSlice } from './slices/authSlice';
import { timerSlice } from './slices/timerSlice';
import { tasksSlice } from './slices/tasksSlice';
import { uiSlice } from './slices/uiSlice';
import { apiSlice } from './api/apiSlice';

export const store = configureStore({
  reducer: {
    auth: authSlice.reducer,
    timer: timerSlice.reducer,
    tasks: tasksSlice.reducer,
    ui: uiSlice.reducer,
    api: apiSlice.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }).concat(apiSlice.middleware),
});

setupListeners(store.dispatch);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

### 3.5 PWA Implementation

**Service Worker Configuration**:
```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\.optopomodoro\.com/,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 60 * 60 * 24 // 24 hours
              }
            }
          }
        ]
      },
      manifest: {
        name: 'OptoPomodoro',
        short_name: 'OptoPomodoro',
        description: 'Zen-inspired productivity for Optomatica teams',
        theme_color: '#7A8B7F',
        background_color: '#F4E4D4',
        display: 'standalone',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      }
    })
  ]
});
```

**Offline-First Data Strategy**:
```typescript
// services/offlineService.ts
import { openDB } from 'idb';

class OfflineService {
  private db: IDBDatabase | null = null;

  async initDB() {
    this.db = await openDB('OptoPomodoroOffline', 1, {
      upgrade(db) {
        db.createObjectStore('sessions', { keyPath: 'id' });
        db.createObjectStore('tasks', { keyPath: 'id' });
        db.createObjectStore('syncQueue', { keyPath: 'id', autoIncrement: true });
      }
    });
  }

  async cacheSession(session: Session) {
    if (!this.db) await this.initDB();
    return this.db!.add('sessions', session);
  }

  async cacheTask(task: Task) {
    if (!this.db) await this.initDB();
    return this.db!.put('tasks', task);
  }

  async queueForSync(action: SyncAction) {
    if (!this.db) await this.initDB();
    return this.db!.add('syncQueue', {
      ...action,
      timestamp: Date.now()
    });
  }

  async getSyncQueue(): Promise<SyncAction[]> {
    if (!this.db) await this.initDB();
    return this.db!.getAll('syncQueue');
  }

  async clearSyncQueue() {
    if (!this.db) await this.initDB();
    return this.db!.clear('syncQueue');
  }
}

export const offlineService = new OfflineService();
```

### 3.6 Design System Implementation

**Theme Configuration**:
```typescript
// themes/ZenTheme.ts
export const ZenTheme = {
  colors: {
    primary: {
      main: '#7A8B7F',      // Moss Green
      light: '#9AA895',
      dark: '#5F6E63',
    },
    secondary: {
      main: '#6B8E9F',      // Water Blue
      light: '#8DA4B1',
      dark: '#537281',
    },
    accent: {
      main: '#E67E50',      // Sunrise Orange
      light: '#EB9D7A',
      dark: '#C46441',
    },
    neutral: {
      50: '#F4E4D4',        // Warm Sand
      100: '#E8D8C8',
      200: '#D4C5B9',      // Zen Stone
      300: '#C0B2A5',
      400: '#8B7D7B',      // Stone Gray
      500: '#2C3E50',      // Charcoal
    },
    success: '#7FA870',     // Sage Green
    warning: '#F4A261',     // Warm Amber
    error: '#C85A5A',       // Soft Red
    info: '#6B8E9F',        // Sky Blue
  },

  typography: {
    fontFamily: {
      primary: 'Inter, sans-serif',
      secondary: 'Lora, serif',
    },
    fontSize: {
      xs: '12px',
      sm: '14px',
      base: '16px',
      lg: '18px',
      xl: '24px',
      '2xl': '32px',
      '3xl': '48px',
    },
    fontWeight: {
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.75,
    },
  },

  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '32px',
    xl: '64px',
  },

  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '16px',
    full: '50%',
  },

  shadows: {
    sm: '0 1px 3px rgba(0, 0, 0, 0.1)',
    md: '0 4px 6px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 25px rgba(0, 0, 0, 0.1)',
  },

  animation: {
    duration: {
      fast: '150ms',
      normal: '300ms',
      slow: '500ms',
    },
    easing: {
      easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    },
  },
};
```

## 4. Backend Architecture (ExpressJS)

### 4.1 Technology Stack

**Framework**: ExpressJS with TypeScript
**Database ORM**: Prisma for type-safe database access
**Authentication**: Passport.js with OAuth 2.0 strategies
**Validation**: Joi/Zod for request validation
**Real-time**: Socket.IO for WebSocket connections
**File Storage**: Multer with local/Cloud storage
**Rate Limiting**: express-rate-limit
**Security**: Helmet, CORS, CSRF protection
**Logging**: Winston for structured logging
**Testing**: Jest with Supertest for API testing

### 4.2 API Architecture

**RESTful API Design**:
```typescript
// routes/index.ts
import express from 'express';
import { authRoutes } from './auth';
import { userRoutes } from './users';
import { taskRoutes } from './tasks';
import { sessionRoutes } from './sessions';
import { teamRoutes } from './teams';
import { analyticsRoutes } from './analytics';
import { gamificationRoutes } from './gamification';
import { integrationRoutes } from './integrations';

export const router = express.Router();

router.use('/auth', authRoutes);
router.use('/users', userRoutes);
router.use('/tasks', taskRoutes);
router.use('/sessions', sessionRoutes);
router.use('/teams', teamRoutes);
router.use('/analytics', analyticsRoutes);
router.use('/gamification', gamificationRoutes);
router.use('/integrations', integrationRoutes);

// Health check endpoint
router.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});
```

**API Response Standardization**:
```typescript
// utils/apiResponse.ts
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  meta?: {
    pagination?: {
      page: number;
      limit: number;
      total: number;
      totalPages: number;
    };
    timestamp: string;
    requestId: string;
  };
}

export const createSuccessResponse = <T>(
  data: T,
  message?: string,
  meta?: any
): ApiResponse<T> => ({
  success: true,
  data,
  message,
  meta: {
    timestamp: new Date().toISOString(),
    requestId: generateRequestId(),
    ...meta,
  },
});

export const createErrorResponse = (
  code: string,
  message: string,
  details?: any
): ApiResponse => ({
  success: false,
  error: {
    code,
    message,
    details,
  },
  meta: {
    timestamp: new Date().toISOString(),
    requestId: generateRequestId(),
  },
});
```

### 4.3 Database Layer (Prisma)

**Schema Definition**:
```prisma
// prisma/schema.prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "sqlite"
  url      = env("DATABASE_URL")
}

model User {
  id          String   @id @default(cuid())
  email       String   @unique
  name        String
  avatar      String?
  preferences Json?
  settings    Json?
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt

  // Relations
  sessions    Session[]
  tasks       Task[]
  teamMembers TeamMember[]
  achievements UserAchievement[]
  notifications Notification[]

  @@map("users")
}

model Team {
  id          String   @id @default(cuid())
  name        String
  description String?
  settings    Json?
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt

  // Relations
  members   TeamMember[]
  tasks     Task[]
  challenges TeamChallenge[]

  @@map("teams")
}

model TeamMember {
  id     String @id @default(cuid())
  userId String
  teamId String
  role   TeamRole @default(MEMBER)
  joinedAt DateTime @default(now())

  // Relations
  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
  team Team @relation(fields: [teamId], references: [id], onDelete: Cascade)

  @@unique([userId, teamId])
  @@map("team_members")
}

model Task {
  id          String     @id @default(cuid())
  title       String
  description String?
  priority    Priority   @default(NORMAL)
  status      TaskStatus @default(TODO)
  dueDate     DateTime?
  estimatedMinutes Int?
  actualMinutes     Int?
  projectId   String?
  assigneeId  String?
  creatorId   String
  teamId      String?
  createdAt   DateTime   @default(now())
  updatedAt   DateTime   @updatedAt

  // Relations
  creator   User     @relation(fields: [creatorId], references: [id])
  assignee  User?    @relation(fields: [assigneeId], references: [id])
  team      Team?    @relation(fields: [teamId], references: [id])
  sessions  Session[]
  dependencies TaskDependency[] @relation("DependentTask")
  dependents   TaskDependency[] @relation("PrerequisiteTask")

  @@map("tasks")
}

model Session {
  id          String      @id @default(cuid())
  userId      String
  taskId      String?
  type        SessionType @default(POMODORO)
  duration    Int         // in minutes
  startedAt   DateTime
  completedAt DateTime?
  notes       String?
  quality     Int?        // 1-5 rating
  createdAt   DateTime    @default(now())

  // Relations
  user User   @relation(fields: [userId], references: [id])
  task Task?  @relation(fields: [taskId], references: [id])

  @@map("sessions")
}

model Achievement {
  id          String @id @default(cuid())
  name        String
  description String
  icon        String
  category    AchievementCategory
  xpValue     Int
  criteria    Json  // Conditions to unlock
  createdAt   DateTime @default(now())

  // Relations
  userAchievements UserAchievement[]

  @@map("achievements")
}

model UserAchievement {
  id           String   @id @default(cuid())
  userId       String
  achievementId String
  unlockedAt   DateTime @default(now())
  progress     Json?    // Progress data for partially completed achievements

  // Relations
  user       User       @relation(fields: [userId], references: [id])
  achievement Achievement @relation(fields: [achievementId], references: [id])

  @@unique([userId, achievementId])
  @@map("user_achievements")
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
```

### 4.4 Authentication & Authorization

**OAuth 2.0 Implementation**:
```typescript
// services/authService.ts
import passport from 'passport';
import { Strategy as GoogleStrategy } from 'passport-google-oauth20';
import { Strategy as MicrosoftStrategy } from 'passport-microsoft';
import { User } from '@prisma/client';
import { prisma } from '../config/database';
import { createToken, verifyToken } from '../utils/jwt';

export class AuthService {
  static initializeStrategies() {
    // Google OAuth Strategy
    passport.use(new GoogleStrategy({
      clientID: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
      callbackURL: "/api/auth/google/callback"
    }, async (accessToken, refreshToken, profile, done) => {
      try {
        const email = profile.emails?.[0]?.value;
        if (!email || !email.endsWith('@optomatica.com')) {
          return done(new Error('Invalid domain'), null);
        }

        let user = await prisma.user.findUnique({
          where: { email }
        });

        if (!user) {
          user = await prisma.user.create({
            data: {
              email,
              name: profile.displayName,
              avatar: profile.photos?.[0]?.value,
              preferences: {},
              settings: {}
            }
          });
        }

        const token = createToken({ userId: user.id, email: user.email });
        return done(null, { user, token });
      } catch (error) {
        return done(error, null);
      }
    }));

    // Microsoft OAuth Strategy
    passport.use(new MicrosoftStrategy({
      clientID: process.env.MICROSOFT_CLIENT_ID!,
      clientSecret: process.env.MICROSOFT_CLIENT_SECRET!,
      callbackURL: "/api/auth/microsoft/callback"
    }, async (accessToken, refreshToken, profile, done) => {
      // Similar implementation as Google strategy
    }));
  }

  static verifyDomain(email: string): boolean {
    return email.endsWith('@optomatica.com');
  }

  static createSession(user: User): { token: string; refreshToken: string } {
    const token = createToken({ userId: user.id, email: user.email });
    const refreshToken = createToken({ userId: user.id }, '7d');

    return { token, refreshToken };
  }
}
```

### 4.5 Real-time Features (WebSocket)

**Socket.IO Implementation**:
```typescript
// websocket/socketHandler.ts
import { Server, Socket } from 'socket.io';
import { verifyToken } from '../utils/jwt';
import { prisma } from '../config/database';

export class SocketHandler {
  private io: Server;

  constructor(server: any) {
    this.io = new Server(server, {
      cors: {
        origin: process.env.FRONTEND_URL,
        credentials: true
      }
    });

    this.initializeMiddleware();
    this.initializeEventHandlers();
  }

  private initializeMiddleware() {
    this.io.use(async (socket, next) => {
      try {
        const token = socket.handshake.auth.token;
        const decoded = verifyToken(token);
        const user = await prisma.user.findUnique({
          where: { id: decoded.userId }
        });

        if (!user) {
          return next(new Error('Authentication failed'));
        }

        socket.data.user = user;
        next();
      } catch (error) {
        next(new Error('Authentication failed'));
      }
    });
  }

  private initializeEventHandlers() {
    this.io.on('connection', (socket) => {
      console.log(`User ${socket.data.user.id} connected`);

      // Join user's personal room
      socket.join(`user:${socket.data.user.id}`);

      // Join team rooms
      this.joinTeamRooms(socket);

      // Handle timer events
      socket.on('timer:start', (data) => this.handleTimerStart(socket, data));
      socket.on('timer:pause', (data) => this.handleTimerPause(socket, data));
      socket.on('timer:complete', (data) => this.handleTimerComplete(socket, data));

      // Handle task events
      socket.on('task:update', (data) => this.handleTaskUpdate(socket, data));
      socket.on('task:complete', (data) => this.handleTaskComplete(socket, data));

      // Handle team events
      socket.on('team:join', (teamId) => this.handleTeamJoin(socket, teamId));
      socket.on('team:leave', (teamId) => this.handleTeamLeave(socket, teamId));

      socket.on('disconnect', () => {
        console.log(`User ${socket.data.user.id} disconnected`);
      });
    });
  }

  private async joinTeamRooms(socket: Socket) {
    const teams = await prisma.teamMember.findMany({
      where: { userId: socket.data.user.id },
      include: { team: true }
    });

    teams.forEach(({ team }) => {
      socket.join(`team:${team.id}`);
    });
  }

  private async handleTimerStart(socket: Socket, data: any) {
    // Broadcast to team members
    const userTeams = await this.getUserTeams(socket.data.user.id);
    userTeams.forEach(teamId => {
      socket.to(`team:${teamId}`).emit('member:timer:start', {
        userId: socket.data.user.id,
        userName: socket.data.user.name,
        ...data
      });
    });
  }

  private async handleTaskComplete(socket: Socket, data: { taskId: string }) {
    const task = await prisma.task.findUnique({
      where: { id: data.taskId },
      include: { team: true }
    });

    if (task?.team) {
      this.io.to(`team:${task.team.id}`).emit('team:task:completed', {
        taskId: task.id,
        userId: socket.data.user.id,
        userName: socket.data.user.name
      });
    }
  }

  private async getUserTeams(userId: string): Promise<string[]> {
    const teams = await prisma.teamMember.findMany({
      where: { userId }
    });
    return teams.map(tm => tm.teamId);
  }
}
```

## 5. Database Design (SQLite)

### 5.1 Schema Optimization

**Indexing Strategy**:
```sql
-- Performance indexes for common queries
CREATE INDEX idx_sessions_user_date ON sessions(user_id, started_at);
CREATE INDEX idx_tasks_assignee_status ON tasks(assignee_id, status);
CREATE INDEX idx_tasks_team_due_date ON tasks(team_id, due_date);
CREATE INDEX idx_user_achievements_user ON user_achievements(user_id);
CREATE INDEX idx_team_members_team ON team_members(team_id);

-- Full-text search for tasks
CREATE VIRTUAL TABLE tasks_fts USING fts5(title, description, content='tasks');

-- Triggers to keep FTS table in sync
CREATE TRIGGER tasks_fts_insert AFTER INSERT ON tasks BEGIN
  INSERT INTO tasks_fts(rowid, title, description) VALUES (new.id, new.title, new.description);
END;

CREATE TRIGGER tasks_fts_delete AFTER DELETE ON tasks BEGIN
  DELETE FROM tasks_fts WHERE rowid = old.id;
END;

CREATE TRIGGER tasks_fts_update AFTER UPDATE ON tasks BEGIN
  DELETE FROM tasks_fts WHERE rowid = old.id;
  INSERT INTO tasks_fts(rowid, title, description) VALUES (new.id, new.title, new.description);
END;
```

**Database Constraints**:
```sql
-- Data integrity constraints
CREATE TRIGGER validate_task_due_date
BEFORE INSERT ON tasks
WHEN NEW.due_date < datetime('now')
BEGIN
  SELECT RAISE(ABORT, 'Due date cannot be in the past');
END;

CREATE TRIGGER validate_session_duration
BEFORE INSERT ON sessions
WHEN NEW.duration < 1 OR NEW.duration > 480
BEGIN
  SELECT RAISE(ABORT, 'Session duration must be between 1 and 480 minutes');
END;

-- Cascade delete handlers
CREATE TRIGGER cleanup_user_data
AFTER DELETE ON users
BEGIN
  DELETE FROM sessions WHERE user_id = old.id;
  DELETE FROM user_achievements WHERE user_id = old.id;
  DELETE FROM team_members WHERE user_id = old.id;
END;
```

### 5.2 Migration Strategy

**Migration Files Structure**:
```typescript
// migrations/001_initial_schema.sql
-- Users table
CREATE TABLE users (
  id TEXT PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  avatar TEXT,
  preferences TEXT, -- JSON
  settings TEXT,    -- JSON
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Teams table
CREATE TABLE teams (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  settings TEXT, -- JSON
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ... other tables
```

**Migration Runner**:
```typescript
// database/migrator.ts
import sqlite3 from 'sqlite3';
import { readFileSync } from 'fs';
import { join } from 'path';

export class DatabaseMigrator {
  private db: sqlite3.Database;

  constructor(dbPath: string) {
    this.db = new sqlite3.Database(dbPath);
  }

  async migrate() {
    // Create migrations table if it doesn't exist
    await this.run(`
      CREATE TABLE IF NOT EXISTS migrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE NOT NULL,
        executed_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Get list of migration files
    const migrationFiles = await this.getMigrationFiles();

    // Get executed migrations
    const executedMigrations = await this.getExecutedMigrations();

    // Run pending migrations
    for (const file of migrationFiles) {
      if (!executedMigrations.includes(file)) {
        await this.runMigration(file);
      }
    }
  }

  private async runMigration(filename: string) {
    const migrationSQL = readFileSync(
      join(__dirname, 'migrations', filename),
      'utf-8'
    );

    console.log(`Running migration: ${filename}`);

    await this.run('BEGIN TRANSACTION');

    try {
      await this.run(migrationSQL);
      await this.run(
        'INSERT INTO migrations (filename) VALUES (?)',
        [filename]
      );
      await this.run('COMMIT');

      console.log(`Migration completed: ${filename}`);
    } catch (error) {
      await this.run('ROLLBACK');
      throw error;
    }
  }

  private async run(sql: string, params: any[] = []): Promise<void> {
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, function(err) {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  private async getMigrationFiles(): Promise<string[]> {
    // Read migration files and sort them
    // Implementation depends on your file system access
    return [];
  }

  private async getExecutedMigrations(): Promise<string[]> {
    return new Promise((resolve, reject) => {
      this.db.all(
        'SELECT filename FROM migrations ORDER BY executed_at',
        (err, rows: any[]) => {
          if (err) reject(err);
          else resolve(rows.map(row => row.filename));
        }
      );
    });
  }
}
```

### 5.3 Performance Optimization

**Connection Pooling**:
```typescript
// database/connection.ts
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';

let dbInstance: any = null;

export const getDatabase = async () => {
  if (!dbInstance) {
    dbInstance = await open({
      filename: process.env.DATABASE_URL || './data/optopomodoro.db',
      driver: sqlite3.Database
    });

    // Configure database for performance
    await dbInstance.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = NORMAL;
      PRAGMA cache_size = 10000;
      PRAGMA temp_store = MEMORY;
      PRAGMA mmap_size = 268435456; -- 256MB
    `);
  }

  return dbInstance;
};
```

**Query Optimization**:
```typescript
// services/analyticsService.ts
export class AnalyticsService {
  // Optimized query for user focus time trends
  async getUserFocusTrends(userId: string, days: number = 30) {
    const db = await getDatabase();

    return db.all(`
      SELECT
        DATE(started_at) as date,
        COUNT(*) as session_count,
        SUM(duration) as total_minutes,
        AVG(quality) as avg_quality
      FROM sessions
      WHERE user_id = ?
        AND started_at >= date('now', '-${days} days')
        AND type = 'POMODORO'
        AND completed_at IS NOT NULL
      GROUP BY DATE(started_at)
      ORDER BY date DESC
    `, [userId]);
  }

  // Optimized team leaderboard query
  async getTeamLeaderboard(teamId: string, period: 'week' | 'month' = 'week') {
    const db = await getDatabase();
    const dateFilter = period === 'week' ? '-7 days' : '-30 days';

    return db.all(`
      SELECT
        u.id,
        u.name,
        u.avatar,
        COUNT(s.id) as session_count,
        SUM(s.duration) as focus_minutes,
        AVG(s.quality) as avg_quality,
        COUNT(DISTINCT DATE(s.started_at)) as active_days
      FROM users u
      INNER JOIN team_members tm ON u.id = tm.user_id
      LEFT JOIN sessions s ON u.id = s.user_id
        AND s.started_at >= date('now', '${dateFilter}')
        AND s.type = 'POMODORO'
        AND s.completed_at IS NOT NULL
      WHERE tm.team_id = ?
      GROUP BY u.id, u.name, u.avatar
      ORDER BY focus_minutes DESC
      LIMIT 10
    `, [teamId]);
  }
}
```

## 6. Feature Implementation Details

### 6.1 Timer System

**Offline-First Timer Implementation**:
```typescript
// services/timerService.ts
export class TimerService {
  private timer: NodeJS.Timeout | null = null;
  private startTime: number | null = null;
  private remainingTime: number = 0;
  private isPaused: boolean = false;
  private currentSession: Partial<Session> | null = null;

  async startSession(config: TimerConfig): Promise<void> {
    this.remainingTime = config.duration * 60; // Convert to seconds
    this.startTime = Date.now();
    this.isPaused = false;

    // Create session record
    this.currentSession = {
      type: config.type,
      duration: config.duration,
      startedAt: new Date(),
      taskId: config.taskId
    };

    // Save to offline storage
    await offlineService.cacheSession(this.currentSession);

    // Start timer
    this.timer = setInterval(() => this.tick(), 1000);

    // Notify server if online
    if (navigator.onLine) {
      await this.syncSessionStart(this.currentSession);
    }
  }

  private async tick(): Promise<void> {
    if (this.isPaused) return;

    this.remainingTime--;

    // Update UI
    this.updateTimerDisplay(this.remainingTime);

    // Check if session complete
    if (this.remainingTime <= 0) {
      await this.completeSession();
    }

    // Sync progress every 30 seconds
    if (this.remainingTime % 30 === 0 && navigator.onLine) {
      await this.syncProgress();
    }
  }

  async completeSession(): Promise<void> {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }

    const completedSession: Session = {
      ...this.currentSession!,
      completedAt: new Date(),
      quality: await this.promptForQuality()
    };

    // Save to offline storage
    await offlineService.cacheSession(completedSession);

    // Queue for sync if offline
    if (!navigator.onLine) {
      await offlineService.queueForSync({
        type: 'SESSION_COMPLETE',
        data: completedSession
      });
    } else {
      await this.syncSessionComplete(completedSession);
    }

    // Trigger celebration animation
    this.triggerCompletionAnimation(completedSession);

    // Update achievements and XP
    await this.updateProgress(completedSession);
  }

  private async syncSessionStart(session: Partial<Session>): Promise<void> {
    try {
      await api.post('/sessions/start', session);
    } catch (error) {
      console.error('Failed to sync session start:', error);
      // Will be queued for retry when online
    }
  }

  private async syncSessionComplete(session: Session): Promise<void> {
    try {
      await api.post('/sessions/complete', session);
    } catch (error) {
      console.error('Failed to sync session complete:', error);
      // Queue for retry
      await offlineService.queueForSync({
        type: 'SESSION_COMPLETE',
        data: session
      });
    }
  }
}
```

### 6.2 Task Management with AI Integration

**Smart Task Suggestions**:
```typescript
// services/taskService.ts
export class TaskService {
  async getSmartTasks(userId: string, context: TaskContext): Promise<TaskSuggestion[]> {
    // Get user's productivity patterns
    const patterns = await this.getProductivityPatterns(userId);

    // Get current energy level (if available)
    const energyLevel = await this.getCurrentEnergyLevel(userId);

    // Get upcoming deadlines
    const deadlines = await this.getUpcomingDeadlines(userId);

    // Generate suggestions
    const suggestions: TaskSuggestion[] = [];

    // High-priority tasks due soon
    deadlines.forEach(deadline => {
      suggestions.push({
        taskId: deadline.id,
        reason: 'Due soon',
        priority: 'HIGH',
        estimatedSessions: this.estimateSessions(deadline, patterns)
      });
    });

    // Energy-appropriate tasks
    if (energyLevel) {
      const energyTasks = await this.getTasksByEnergyRequirement(userId, energyLevel);
      energyTasks.forEach(task => {
        suggestions.push({
          taskId: task.id,
          reason: `Matches your ${energyLevel} energy level`,
          priority: 'MEDIUM',
          estimatedSessions: this.estimateSessions(task, patterns)
        });
      });
    }

    // Learning-based recommendations
    const learnedSuggestions = await this.getLearnedSuggestions(userId, patterns);
    suggestions.push(...learnedSuggestions);

    return this.rankSuggestions(suggestions);
  }

  private async getProductivityPatterns(userId: string): Promise<ProductivityPattern> {
    const analytics = await api.get(`/analytics/patterns/${userId}`);

    return {
      peakHours: analytics.peakHours,
      averageSessionLength: analytics.averageSessionLength,
      taskCompletionRates: analytics.taskCompletionRates,
      preferredBreakIntervals: analytics.preferredBreakIntervals
    };
  }

  private estimateSessions(task: Task, patterns: ProductivityPattern): number {
    // Base estimation on task complexity
    let sessions = Math.ceil(task.estimatedMinutes / 25);

    // Adjust based on user patterns
    if (task.priority === 'CRITICAL') {
      sessions = Math.ceil(sessions * 1.2);
    }

    // Adjust for task type
    if (task.category === 'CREATIVE') {
      sessions = Math.ceil(sessions * 1.5); // Creative tasks take longer
    }

    return sessions;
  }
}
```

### 6.3 Gamification System

**Achievement Engine**:
```typescript
// services/achievementService.ts
export class AchievementService {
  async checkAchievements(userId: string, event: AchievementEvent): Promise<Achievement[]> {
    const userAchievements = await this.getUserAchievements(userId);
    const allAchievements = await this.getAllAchievements();

    const newAchievements: Achievement[] = [];

    for (const achievement of allAchievements) {
      // Skip if already unlocked
      if (userAchievements.find(ua => ua.achievementId === achievement.id)) {
        continue;
      }

      // Check if criteria are met
      if (await this.checkCriteria(achievement, userId, event)) {
        await this.unlockAchievement(userId, achievement.id);
        newAchievements.push(achievement);
      }
    }

    return newAchievements;
  }

  private async checkCriteria(
    achievement: Achievement,
    userId: string,
    event: AchievementEvent
  ): Promise<boolean> {
    const criteria = achievement.criteria as AchievementCriteria;

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

  private async checkSessionCount(
    userId: string,
    criteria: SessionCountCriteria
  ): Promise<boolean> {
    const sessions = await api.get(`/sessions/count/${userId}`, {
      params: {
        type: criteria.sessionType,
        timeRange: criteria.timeRange
      }
    });

    return sessions.count >= criteria.requiredCount;
  }

  private async checkConsecutiveDays(
    userId: string,
    criteria: ConsecutiveDaysCriteria
  ): Promise<boolean> {
    const streaks = await api.get(`/analytics/streaks/${userId}`);
    return streaks.currentStreak >= criteria.requiredDays;
  }

  async calculateXp(userId: string, event: AchievementEvent): Promise<number> {
    let xp = 0;

    switch (event.type) {
      case 'SESSION_COMPLETE':
        xp = 10; // Base XP for session
        if (event.data.quality >= 4) xp += 5; // Quality bonus
        if (event.data.duration >= 50) xp += 5; // Deep work bonus
        break;

      case 'TASK_COMPLETE':
        const complexity = event.data.complexity || 1;
        xp = 5 * complexity; // XP based on task complexity

        // Deadline bonus
        if (event.data.completedEarly) {
          xp = Math.ceil(xp * 1.5);
        }
        break;

      case 'STREAK_MILESTONE':
        xp = event.data.streakDays * 2;
        break;

      case 'TEAM_CHALLENGE_COMPLETE':
        xp = 25; // Team participation bonus
        break;
    }

    return xp;
  }
}
```

### 6.4 Real-time Team Collaboration

**Live Team Features**:
```typescript
// services/teamService.ts
export class TeamService {
  private socket: Socket;

  constructor(socket: Socket) {
    this.socket = socket;
    this.initializeTeamEvents();
  }

  private initializeTeamEvents(): void {
    // Live team status updates
    this.socket.on('team:member:status', (data) => {
      this.updateTeamMemberStatus(data);
    });

    // Team challenges
    this.socket.on('team:challenge:start', (challenge) => {
      this.handleTeamChallengeStart(challenge);
    });

    // Real-time recognition
    this.socket.on('team:recognition', (recognition) => {
      this.handleRecognition(recognition);
    });

    // Team focus sessions
    this.socket.on('team:focus:session', (session) => {
      this.handleTeamFocusSession(session);
    });
  }

  async startTeamChallenge(teamId: string, challengeType: TeamChallengeType): Promise<void> {
    const challenge = await api.post('/teams/challenges/start', {
      teamId,
      type: challengeType,
      startDate: new Date(),
      endDate: this.calculateChallengeEndDate(challengeType)
    });

    // Notify team members
    this.socket.emit('team:challenge:start', challenge.data);
  }

  async sendRecognition(teamId: string, recipientId: string, message: string): Promise<void> {
    const recognition = await api.post('/teams/recognition', {
      teamId,
      recipientId,
      senderId: this.getCurrentUserId(),
      message,
      timestamp: new Date()
    });

    // Broadcast to team
    this.socket.to(`team:${teamId}`).emit('team:recognition', recognition.data);
  }

  async joinTeamFocusSession(teamId: string, sessionId: string): Promise<void> {
    await api.post(`/teams/${teamId}/focus-sessions/${sessionId}/join`);

    // Update presence
    this.socket.emit('team:focus:join', {
      teamId,
      sessionId,
      userId: this.getCurrentUserId()
    });
  }

  private calculateChallengeEndDate(challengeType: TeamChallengeType): Date {
    const now = new Date();
    switch (challengeType) {
      case 'DAILY_FOCUS':
        return new Date(now.getTime() + 24 * 60 * 60 * 1000);
      case 'WEEKLY_PRODUCTIVITY':
        return new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);
      case 'MONTHLY_WELLNESS':
        return new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000);
      default:
        return new Date(now.getTime() + 24 * 60 * 60 * 1000);
    }
  }
}
```

## 7. Integration Specifications

### 7.1 Calendar Integration

**Google Calendar Integration**:
```typescript
// services/calendarService.ts
export class CalendarService {
  private googleCalendar: calendar_v3.Calendar;

  constructor() {
    const auth = new google.auth.OAuth2(
      process.env.GOOGLE_CLIENT_ID,
      process.env.GOOGLE_CLIENT_SECRET,
      process.env.GOOGLE_REDIRECT_URI
    );

    this.googleCalendar = google.calendar({ version: 'v3', auth });
  }

  async syncWithGoogleCalendar(userId: string, tokens: any): Promise<void> {
    // Set up authentication
    this.googleCalendar.context.auth.setCredentials(tokens);

    // Get user's calendar events
    const events = await this.googleCalendar.events.list({
      calendarId: 'primary',
      timeMin: new Date().toISOString(),
      timeMax: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      singleEvents: true,
      orderBy: 'startTime'
    });

    // Process events and create tasks
    for (const event of events.data.items || []) {
      if (this.shouldCreateTaskFromEvent(event)) {
        await this.createTaskFromEvent(userId, event);
      }
    }

    // Set up webhook for real-time updates
    await this.setupGoogleCalendarWebhook(userId, tokens);
  }

  private shouldCreateTaskFromEvent(event: calendar_v3.Schema$Event): boolean {
    // Only create tasks for events with specific keywords
    const taskKeywords = ['deadline', 'review', 'complete', 'finish'];
    const title = event.summary?.toLowerCase() || '';

    return taskKeywords.some(keyword => title.includes(keyword)) ||
           event.description?.toLowerCase().includes('deadline') ||
           false;
  }

  private async createTaskFromEvent(userId: string, event: calendar_v3.Schema$Event): Promise<void> {
    const task = {
      title: event.summary || 'Calendar Task',
      description: event.description || '',
      dueDate: event.end?.dateTime || event.end?.date,
      creatorId: userId,
      priority: this.determinePriorityFromEvent(event),
      estimatedMinutes: this.estimateDurationFromEvent(event)
    };

    await api.post('/tasks', task);
  }

  async blockFocusTime(userId: string, sessions: FocusSession[]): Promise<void> {
    for (const session of sessions) {
      const event = {
        summary: 'Focus Time - OptoPomodoro',
        description: `Deep work session - ${session.duration} minutes`,
        start: {
          dateTime: session.startTime.toISOString(),
          timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
        },
        end: {
          dateTime: session.endTime.toISOString(),
          timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
        },
        colorId: '2' // Green color for focus time
      };

      await this.googleCalendar.events.insert({
        calendarId: 'primary',
        requestBody: event
      });
    }
  }
}
```

### 7.2 Communication Tool Integration

**Slack Integration**:
```typescript
// services/slackService.ts
export class SlackService {
  private slackClient: WebClient;

  constructor(botToken: string) {
    this.slackClient = new WebClient(botToken);
  }

  async updateSlackStatus(userId: string, status: FocusStatus): Promise<void> {
    const statusEmoji = this.getStatusEmoji(status);
    const statusText = this.getStatusText(status);

    await this.slackClient.users.profile.set({
      profile: {
        status_text: statusText,
        status_emoji: statusEmoji,
        status_expiration: status.expiresAt
      }
    });
  }

  private getStatusEmoji(status: FocusStatus): string {
    switch (status.type) {
      case 'FOCUS': return ':tomato:';
      case 'BREAK': return ':coffee:';
      case 'DEEP_WORK': return ':brain:';
      case 'MEETING': return ':calendar:';
      default: return ':house:';
    }
  }

  private getStatusText(status: FocusStatus): string {
    switch (status.type) {
      case 'FOCUS': return `In focus mode (${status.remainingMinutes}m left)`;
      case 'BREAK': return `Taking a break (${status.remainingMinutes}m left)`;
      case 'DEEP_WORK': return 'Deep work - please interrupt only if urgent';
      case 'MEETING': return 'In a meeting';
      default: return 'Available';
    }
  }

  async shareAchievement(userId: string, achievement: Achievement, teamChannel?: string): Promise<void> {
    const message = {
      text: `🎉 Congratulations to <@${userId}> for unlocking the *${achievement.name}* achievement!`,
      blocks: [
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `🎉 *Achievement Unlocked!*`
          }
        },
        {
          type: 'section',
          fields: [
            {
              type: 'mrkdwn',
              text: `*User:*\n<@${userId}>`
            },
            {
              type: 'mrkdwn',
              text: `*Achievement:*\n${achievement.icon} ${achievement.name}`
            }
          ]
        },
        {
          type: 'section',
          text: {
            type: 'plain_text',
            text: achievement.description
          }
        }
      ]
    };

    // Post to user's DM
    await this.slackClient.chat.postMessage({
      channel: userId,
      ...message
    });

    // Post to team channel if specified
    if (teamChannel) {
      await this.slackClient.chat.postMessage({
        channel: teamChannel,
        ...message
      });
    }
  }

  async setupDailyDigest(teamId: string, channel: string): Promise<void> {
    // Schedule daily digest message
    const cronJob = new CronJob('0 9 * * *', async () => {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);

      const teamStats = await api.get(`/teams/${teamId}/stats`, {
        params: { date: yesterday.toISOString().split('T')[0] }
      });

      const message = {
        text: `📊 Yesterday's Team Productivity Report`,
        blocks: [
          {
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: `📊 *Team Productivity Report - ${yesterday.toLocaleDateString()}*`
            }
          },
          {
            type: 'section',
            fields: [
              {
                type: 'mrkdwn',
                text: `*Total Focus Time:*\n${teamStats.totalFocusMinutes} minutes`
              },
              {
                type: 'mrkdwn',
                text: `*Tasks Completed:*\n${teamStats.tasksCompleted}`
              },
              {
                type: 'mrkdwn',
                text: `*Team Members Active:*\n${teamStats.activeMembers}`
              },
              {
                type: 'mrkdwn',
                text: `*Average Session Quality:*\n${teamStats.averageQuality}/5`
              }
            ]
          }
        ]
      };

      await this.slackClient.chat.postMessage({
        channel,
        ...message
      });
    });

    cronJob.start();
  }
}
```

## 8. Security & Performance

### 8.1 Security Implementation

**Authentication & Authorization**:
```typescript
// middleware/auth.ts
export const authenticateToken = (req: Request, res: Response, next: NextFunction) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json(createErrorResponse('MISSING_TOKEN', 'Access token required'));
  }

  try {
    const decoded = verifyToken(token);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(403).json(createErrorResponse('INVALID_TOKEN', 'Invalid or expired token'));
  }
};

export const requireOptomaticaDomain = (req: Request, res: Response, next: NextFunction) => {
  if (!req.user.email.endsWith('@optomatica.com')) {
    return res.status(403).json(createErrorResponse('INVALID_DOMAIN', 'Access restricted to Optomatica employees'));
  }
  next();
};

export const requireTeamRole = (roles: TeamRole[]) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    const teamId = req.params.teamId;
    const userId = req.user.userId;

    const membership = await prisma.teamMember.findUnique({
      where: {
        userId_teamId: {
          userId,
          teamId
        }
      }
    });

    if (!membership || !roles.includes(membership.role)) {
      return res.status(403).json(createErrorResponse('INSUFFICIENT_PERMISSIONS', 'Insufficient team permissions'));
    }

    req.teamRole = membership.role;
    next();
  };
};
```

**Data Encryption**:
```typescript
// utils/encryption.ts
import crypto from 'crypto';

export class EncryptionService {
  private static algorithm = 'aes-256-gcm';
  private static keyLength = 32;

  static encrypt(text: string, key: string): { encrypted: string; iv: string; tag: string } {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher(this.algorithm, key);

    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');

    const tag = cipher.getAuthTag();

    return {
      encrypted,
      iv: iv.toString('hex'),
      tag: tag.toString('hex')
    };
  }

  static decrypt(encryptedData: { encrypted: string; iv: string; tag: string }, key: string): string {
    const decipher = crypto.createDecipher(this.algorithm, key);
    decipher.setAuthTag(Buffer.from(encryptedData.tag, 'hex'));

    let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');

    return decrypted;
  }

  static hashPassword(password: string): { hash: string; salt: string } {
    const salt = crypto.randomBytes(16).toString('hex');
    const hash = crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha512').toString('hex');

    return { hash, salt };
  }

  static verifyPassword(password: string, hash: string, salt: string): boolean {
    const verifyHash = crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha512').toString('hex');
    return hash === verifyHash;
  }
}
```

### 8.2 Performance Optimization

**Caching Strategy**:
```typescript
// services/cacheService.ts
import NodeCache from 'node-cache';

export class CacheService {
  private static instance: NodeCache;

  static getInstance(): NodeCache {
    if (!CacheService.instance) {
      CacheService.instance = new NodeCache({
        stdTTL: 600, // 10 minutes default TTL
        checkperiod: 120, // Check for expired keys every 2 minutes
        useClones: false
      });
    }
    return CacheService.instance;
  }

  static async get<T>(key: string): Promise<T | null> {
    const cache = CacheService.getInstance();
    const value = cache.get<T>(key);
    return value || null;
  }

  static async set<T>(key: string, value: T, ttl?: number): Promise<void> {
    const cache = CacheService.getInstance();
    cache.set(key, value, ttl);
  }

  static async invalidate(pattern: string): Promise<void> {
    const cache = CacheService.getInstance();
    const keys = cache.keys();
    const regex = new RegExp(pattern);

    keys.forEach(key => {
      if (regex.test(key)) {
        cache.del(key);
      }
    });
  }

  // Cache warming for frequently accessed data
  static async warmCache(userId: string): Promise<void> {
    // Cache user's recent tasks
    const recentTasks = await prisma.task.findMany({
      where: {
        OR: [
          { creatorId: userId },
          { assigneeId: userId }
        ],
        take: 50,
        orderBy: { updatedAt: 'desc' }
      }
    });

    await this.set(`user:${userId}:recent_tasks`, recentTasks, 300); // 5 minutes

    // Cache user's achievements
    const achievements = await prisma.userAchievement.findMany({
      where: { userId },
      include: { achievement: true }
    });

    await this.set(`user:${userId}:achievements`, achievements, 1800); // 30 minutes

    // Cache today's sessions
    const today = new Date().toISOString().split('T')[0];
    const todaySessions = await prisma.session.findMany({
      where: {
        userId,
        startedAt: {
          gte: new Date(today)
        }
      }
    });

    await this.set(`user:${userId}:sessions:${today}`, todaySessions, 600); // 10 minutes
  }
}
```

**API Rate Limiting**:
```typescript
// middleware/rateLimit.ts
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);

export const createRateLimit = (windowMs: number, max: number, message?: string) => {
  return rateLimit({
    store: new RedisStore({
      sendCommand: (...args: string[]) => redis.call(...args),
    }),
    windowMs,
    max,
    message: {
      success: false,
      error: {
        code: 'RATE_LIMIT_EXCEEDED',
        message: message || 'Too many requests, please try again later'
      }
    },
    standardHeaders: true,
    legacyHeaders: false,
  });
};

// Different rate limits for different endpoints
export const authRateLimit = createRateLimit(
  15 * 60 * 1000, // 15 minutes
  5, // 5 attempts
  'Too many authentication attempts, please try again later'
);

export const apiRateLimit = createRateLimit(
  15 * 60 * 1000, // 15 minutes
  1000, // 1000 requests
  'API rate limit exceeded'
);

export const uploadRateLimit = createRateLimit(
  60 * 60 * 1000, // 1 hour
  10, // 10 uploads
  'Upload rate limit exceeded'
);
```

## 9. Development & Deployment

### 9.1 Development Environment Setup

**Docker Development Environment**:
```dockerfile
# docker-compose.dev.yml
version: '3.8'

services:
  frontend:
    build:
      context: .
      dockerfile: docker/frontend.Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - ./packages/frontend:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:3001
      - VITE_WS_URL=ws://localhost:3001
    depends_on:
      - backend

  backend:
    build:
      context: .
      dockerfile: docker/backend.Dockerfile
    ports:
      - "3001:3001"
    volumes:
      - ./packages/backend:/app
      - /app/node_modules
      - ./data:/app/data
    environment:
      - NODE_ENV=development
      - DATABASE_URL=file:./data/optopomodoro.db
      - JWT_SECRET=dev-secret-key
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

**Package.json Scripts**:
```json
{
  "scripts": {
    "dev": "concurrently \"pnpm --filter frontend dev\" \"pnpm --filter backend dev\"",
    "build": "turbo run build",
    "test": "turbo run test",
    "test:e2e": "pnpm --filter frontend test:e2e",
    "lint": "turbo run lint",
    "lint:fix": "turbo run lint:fix",
    "type-check": "turbo run type-check",
    "db:migrate": "pnpm --filter database migrate",
    "db:seed": "pnpm --filter database seed",
    "docker:dev": "docker-compose -f docker-compose.dev.yml up",
    "docker:build": "docker-compose -f docker-compose.prod.yml build",
    "deploy": "./scripts/deploy.sh"
  }
}
```

### 9.2 Testing Strategy

**Unit Testing (Vitest)**:
```typescript
// packages/frontend/src/components/Timer/__tests__/Timer.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { Timer } from '../Timer';
import { store } from '../../../store';

describe('Timer Component', () => {
  it('should display initial time correctly', () => {
    render(
      <Provider store={store}>
        <Timer initialMinutes={25} />
      </Provider>
    );

    expect(screen.getByText('25:00')).toBeInTheDocument();
  });

  it('should start timer when start button is clicked', async () => {
    render(
      <Provider store={store}>
        <Timer initialMinutes={25} />
      </Provider>
    );

    const startButton = screen.getByRole('button', { name: /start/i });
    fireEvent.click(startButton);

    await waitFor(() => {
      expect(screen.getByText('24:59')).toBeInTheDocument();
    });
  });

  it('should pause timer when pause button is clicked', async () => {
    render(
      <Provider store={store}>
        <Timer initialMinutes={25} />
      </Provider>
    );

    const startButton = screen.getByRole('button', { name: /start/i });
    fireEvent.click(startButton);

    await waitFor(() => {
      expect(screen.getByText('24:59')).toBeInTheDocument();
    });

    const pauseButton = screen.getByRole('button', { name: /pause/i });
    fireEvent.click(pauseButton);

    // Verify timer is paused
    await waitFor(() => {
      expect(screen.getByText('24:59')).toBeInTheDocument();
    }, { timeout: 2000 });
  });
});
```

**Integration Testing (Supertest)**:
```typescript
// packages/backend/src/__tests__/auth.test.ts
import request from 'supertest';
import { app } from '../app';
import { prisma } from '../config/database';

describe('Authentication Endpoints', () => {
  beforeEach(async () => {
    // Clean up database
    await prisma.user.deleteMany();
  });

  describe('POST /api/auth/register', () => {
    it('should register a new user with valid Optomatica email', async () => {
      const userData = {
        email: 'test@optomatica.com',
        name: 'Test User',
        password: 'SecurePassword123!'
      };

      const response = await request(app)
        .post('/api/auth/register')
        .send(userData)
        .expect(201);

      expect(response.body.success).toBe(true);
      expect(response.body.data.user.email).toBe(userData.email);
      expect(response.body.data.token).toBeDefined();
    });

    it('should reject registration with non-Optomatica email', async () => {
      const userData = {
        email: 'test@gmail.com',
        name: 'Test User',
        password: 'SecurePassword123!'
      };

      const response = await request(app)
        .post('/api/auth/register')
        .send(userData)
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('INVALID_DOMAIN');
    });
  });

  describe('POST /api/auth/login', () => {
    beforeEach(async () => {
      // Create a test user
      await prisma.user.create({
        data: {
          email: 'test@optomatica.com',
          name: 'Test User',
          password: 'hashedPassword'
        }
      });
    });

    it('should login with valid credentials', async () => {
      const loginData = {
        email: 'test@optomatica.com',
        password: 'SecurePassword123!'
      };

      const response = await request(app)
        .post('/api/auth/login')
        .send(loginData)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data.token).toBeDefined();
    });

    it('should reject login with invalid credentials', async () => {
      const loginData = {
        email: 'test@optomatica.com',
        password: 'WrongPassword'
      };

      const response = await request(app)
        .post('/api/auth/login')
        .send(loginData)
        .expect(401);

      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('INVALID_CREDENTIALS');
    });
  });
});
```

**E2E Testing (Cypress)**:
```typescript
// cypress/e2e/pomodoro-workflow.cy.ts
describe('Pomodoro Workflow', () => {
  beforeEach(() => {
    // Login and set up test data
    cy.login('test@optomatica.com');
    cy.visit('/');
  });

  it('should complete a full Pomodoro cycle', () => {
    // Start a Pomodoro session
    cy.get('[data-testid="timer-display"]').should('contain', '25:00');
    cy.get('[data-testid="start-button"]').click();

    // Verify timer is running
    cy.get('[data-testid="timer-display"]').should('not.contain', '25:00');
    cy.get('[data-testid="pause-button"]').should('be.visible');

    // Fast-forward timer (for testing)
    cy.clock();
    cy.tick(25 * 60 * 1000);

    // Verify session completion
    cy.get('[data-testid="completion-modal"]').should('be.visible');
    cy.get('[data-testid="quality-rating"]').should('be.visible');

    // Rate session quality
    cy.get('[data-testid="quality-5"]').click();
    cy.get('[data-testid="complete-session"]').click();

    // Verify achievement notification
    cy.get('[data-testid="achievement-notification"]').should('be.visible');

    // Start break automatically
    cy.get('[data-testid="timer-display"]').should('contain', '5:00');
    cy.get('[data-testid="break-indicator"]').should('be.visible');
  });

  it('should track task progress during Pomodoro session', () => {
    // Create a task
    cy.get('[data-testid="add-task-button"]').click();
    cy.get('[data-testid="task-title-input"]').type('Test Task');
    cy.get('[data-testid="save-task-button"]').click();

    // Start timer with task
    cy.get('[data-testid="task-card"]').first().click();
    cy.get('[data-testid="start-button"]').click();

    // Complete session
    cy.clock();
    cy.tick(25 * 60 * 1000);
    cy.get('[data-testid="complete-session"]').click();

    // Verify task progress updated
    cy.get('[data-testid="task-card"]').first()
      .should('contain', 'Pomodoro sessions: 1');
  });
});
```

### 9.3 CI/CD Pipeline

**GitHub Actions Workflow**:
```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'pnpm'

      - name: Install pnpm
        run: npm install -g pnpm

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Type check
        run: pnpm type-check

      - name: Lint
        run: pnpm lint

      - name: Run tests
        run: pnpm test
        env:
          NODE_ENV: test
          DATABASE_URL: file:./test.db
          REDIS_URL: redis://localhost:6379

      - name: Run E2E tests
        run: pnpm test:e2e
        env:
          CYPRESS_baseUrl: http://localhost:3000

      - name: Build
        run: pnpm build

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage/lcov.info

  deploy-staging:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'

    steps:
      - uses: actions/checkout@v3

      - name: Deploy to Staging
        run: |
          echo "Deploying to staging environment"
          # Add deployment commands here

  deploy-production:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Deploy to Production
        run: |
          echo "Deploying to production environment"
          # Add deployment commands here

      - name: Run smoke tests
        run: |
          echo "Running smoke tests"
          # Add smoke test commands here
```

## 10. Migration & Scaling Strategy

### 10.1 Database Migration Path

**SQLite to PostgreSQL Migration**:
```typescript
// scripts/migrate-to-postgres.ts
import { PrismaClient } from '@prisma/client';
import { Pool } from 'pg';

export class DatabaseMigrator {
  private sqlitePrisma: PrismaClient;
  private postgresPool: Pool;

  constructor() {
    this.sqlitePrisma = new PrismaClient({
      datasources: {
        db: {
          url: 'file:./data/optopomodoro.db'
        }
      }
    });

    this.postgresPool = new Pool({
      connectionString: process.env.POSTGRES_URL
    });
  }

  async migrateAllData(): Promise<void> {
    console.log('Starting database migration...');

    try {
      // Migrate in order of dependencies
      await this.migrateUsers();
      await this.migrateTeams();
      await this.migrateTasks();
      await this.migrateSessions();
      await this.migrateAchievements();
      await this.migrateUserAchievements();

      console.log('Migration completed successfully');
    } catch (error) {
      console.error('Migration failed:', error);
      throw error;
    }
  }

  private async migrateUsers(): Promise<void> {
    console.log('Migrating users...');

    const users = await this.sqlitePrisma.user.findMany();

    for (const user of users) {
      await this.postgresPool.query(`
        INSERT INTO users (
          id, email, name, avatar, preferences, settings,
          created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (id) DO UPDATE SET
          email = EXCLUDED.email,
          name = EXCLUDED.name,
          avatar = EXCLUDED.avatar,
          preferences = EXCLUDED.preferences,
          settings = EXCLUDED.settings,
          updated_at = EXCLUDED.updated_at
      `, [
        user.id,
        user.email,
        user.name,
        user.avatar,
        JSON.stringify(user.preferences),
        JSON.stringify(user.settings),
        user.createdAt,
        user.updatedAt
      ]);
    }

    console.log(`Migrated ${users.length} users`);
  }

  // Similar methods for other tables...
}
```

**Prisma Schema for PostgreSQL**:
```prisma
// prisma/schema-postgres.prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("POSTGRES_URL")
}

// Same models as SQLite schema but with PostgreSQL-specific optimizations
model User {
  id          String   @id @default(cuid())
  email       String   @unique
  name        String
  avatar      String?
  preferences Json     @default("{}")
  settings    Json     @default("{}")
  createdAt   DateTime @default(now()) @map("created_at")
  updatedAt   DateTime @updatedAt @map("updated_at")

  // Enhanced indexes for PostgreSQL
  @@index([email])
  @@index([createdAt])
  @@index([updatedAt])
  @@map("users")
}

// Additional optimizations for team collaboration queries
model TeamMember {
  id     String @id @default(cuid())
  userId String @map("user_id")
  teamId String @map("team_id")
  role   TeamRole @default(MEMBER)
  joinedAt DateTime @default(now()) @map("joined_at")

  // Relations
  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
  team Team @relation(fields: [teamId], references: [id], onDelete: Cascade)

  // Optimized indexes
  @@unique([userId, teamId])
  @@index([teamId])
  @@index([userId])
  @@map("team_members")
}
```

### 10.2 Scaling Strategy

**Horizontal Scaling Architecture**:
```typescript
// infrastructure/scaling-config.ts
export class ScalingConfig {
  // Database connection pooling for high concurrency
  static getDatabaseConfig() {
    const isProduction = process.env.NODE_ENV === 'production';

    if (isProduction) {
      return {
        datasource: {
          db: {
            url: process.env.POSTGRES_URL,
            // Connection pooling for PostgreSQL
            connection_limit: 20,
            pool_timeout: 30,
            connect_timeout: 60
          }
        }
      };
    }

    // SQLite configuration for development
    return {
      datasource: {
        db: {
          url: 'file:./data/optopomodoro.db'
        }
      }
    };
  }

  // Redis clustering for session storage
  static getRedisConfig() {
    if (process.env.NODE_ENV === 'production') {
      return {
        host: process.env.REDIS_HOST,
        port: parseInt(process.env.REDIS_PORT || '6379'),
        // Cluster configuration for high availability
        cluster: [
          { host: 'redis-1', port: 6379 },
          { host: 'redis-2', port: 6379 },
          { host: 'redis-3', port: 6379 }
        ],
        enableOfflineQueue: false,
        retryDelayOnFailover: 100,
        maxRetriesPerRequest: 3
      };
    }

    return { host: 'localhost', port: 6379 };
  }

  // Load balancing configuration
  static getLoadBalancerConfig() {
    return {
      algorithm: 'round-robin',
      healthCheck: {
        interval: 30000,
        timeout: 5000,
        retries: 3
      },
      sessionAffinity: {
        enabled: true,
        timeout: 300000 // 5 minutes
      }
    };
  }
}
```

**Microservices Transition Path**:
```typescript
// Future microservices architecture plan
/*
Phase 1: Monolith with clear service boundaries
├── Auth Service (within monolith)
├── Timer Service (within monolith)
├── Task Service (within monolith)
├── Analytics Service (within monolith)
└── Notification Service (within monolith)

Phase 2: Extract external services
├── Auth Service (separate microservice)
├── Timer Service (within monolith)
├── Task Service (within monolith)
├── Analytics Service (separate microservice)
└── Notification Service (separate microservice)

Phase 3: Full microservices
├── Auth Service
├── Timer Service
├── Task Service
├── Analytics Service
├── Notification Service
├── Team Service
└── Integration Service

Each service will have:
- Own database instance
- API Gateway integration
- Service discovery
- Circuit breakers
- Distributed tracing
*/
```

### 10.3 Monitoring & Observability

**Application Monitoring**:
```typescript
// monitoring/monitoringService.ts
import { createPrometheusMetrics } from './prometheus';
import { logger } from './logger';

export class MonitoringService {
  private static metrics = createPrometheusMetrics();

  // Custom metrics for business logic
  static recordSessionStart(userId: string, sessionType: string): void {
    this.metrics.sessionStarts.inc({
      user_id: userId,
      session_type: sessionType
    });

    logger.info('Session started', { userId, sessionType });
  }

  static recordSessionComplete(
    userId: string,
    sessionType: string,
    duration: number,
    quality: number
  ): void {
    this.metrics.sessionCompletions.inc({
      user_id: userId,
      session_type: sessionType
    });

    this.metrics.sessionDuration.observe({
      session_type: sessionType
    }, duration);

    this.metrics.sessionQuality.set({
      user_id: userId
    }, quality);

    logger.info('Session completed', {
      userId,
      sessionType,
      duration,
      quality
    });
  }

  static recordTaskComplete(userId: string, taskComplexity: number): void {
    this.metrics.taskCompletions.inc({
      user_id: userId,
      complexity: taskComplexity.toString()
    });

    logger.info('Task completed', { userId, taskComplexity });
  }

  // Performance metrics
  static recordApiRequest(method: string, route: string, statusCode: number, duration: number): void {
    this.metrics.apiRequestDuration.observe({
      method,
      route,
      status_code: statusCode.toString()
    }, duration);

    this.metrics.apiRequestCount.inc({
      method,
      route,
      status_code: statusCode.toString()
    });
  }

  static recordDatabaseQuery(operation: string, table: string, duration: number): void {
    this.metrics.dbQueryDuration.observe({
      operation,
      table
    }, duration);
  }

  // Health checks
  static async performHealthChecks(): Promise<HealthCheckResult> {
    const checks = await Promise.allSettled([
      this.checkDatabase(),
      this.checkRedis(),
      this.checkExternalServices()
    ]);

    const results = checks.map(check =>
      check.status === 'fulfilled' ? check.value : { status: 'unhealthy', error: check.reason }
    );

    const overallHealth = results.every(r => r.status === 'healthy') ? 'healthy' : 'unhealthy';

    return {
      status: overallHealth,
      timestamp: new Date().toISOString(),
      checks: results
    };
  }

  private static async checkDatabase(): Promise<HealthCheck> {
    try {
      const start = Date.now();
      await prisma.$queryRaw`SELECT 1`;
      const duration = Date.now() - start;

      return {
        status: duration < 1000 ? 'healthy' : 'degraded',
        duration,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  private static async checkRedis(): Promise<HealthCheck> {
    // Similar Redis health check implementation
  }

  private static async checkExternalServices(): Promise<HealthCheck> {
    // Check external service availability
  }
}
```

**Logging Strategy**:
```typescript
// monitoring/logger.ts
import winston from 'winston';
import 'winston-daily-rotate-file';

export const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: {
    service: 'optopomodoro',
    version: process.env.APP_VERSION
  },
  transports: [
    // Error log file
    new winston.transports.DailyRotateFile({
      filename: 'logs/error-%DATE%.log',
      datePattern: 'YYYY-MM-DD',
      level: 'error',
      maxSize: '20m',
      maxFiles: '14d'
    }),

    // Combined log file
    new winston.transports.DailyRotateFile({
      filename: 'logs/combined-%DATE%.log',
      datePattern: 'YYYY-MM-DD',
      maxSize: '20m',
      maxFiles: '14d'
    }),

    // Console output for development
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ],

  // Exception handling
  exceptionHandlers: [
    new winston.transports.File({ filename: 'logs/exceptions.log' })
  ],

  // Rejection handling
  rejectionHandlers: [
    new winston.transports.File({ filename: 'logs/rejections.log' })
  ]
});

// Structured logging helpers
export const logSessionEvent = (event: SessionEvent) => {
  logger.info('Session event', {
    event_type: event.type,
    user_id: event.userId,
    session_id: event.sessionId,
    duration: event.duration,
    quality: event.quality,
    timestamp: event.timestamp
  });
};

export const logApiError = (error: ApiError) => {
  logger.error('API error', {
    error_code: error.code,
    message: error.message,
    stack: error.stack,
    request_id: error.requestId,
    user_id: error.userId,
    endpoint: error.endpoint,
    method: error.method,
    timestamp: new Date().toISOString()
  });
};
```

## Conclusion

This Technical Specification provides a comprehensive roadmap for implementing OptoPomodoro using ReactJS, ExpressJS, and SQLite in a monorepo architecture. The design prioritizes:

1. **Scalability**: Clear migration paths from SQLite to PostgreSQL and monolith to microservices
2. **Performance**: Optimized queries, caching strategies, and PWA capabilities
3. **Security**: Enterprise-grade authentication, data encryption, and privacy controls
4. **Maintainability**: Clean architecture, comprehensive testing, and clear separation of concerns
5. **User Experience**: Offline-first functionality, real-time collaboration, and delightful interactions

The architecture is designed to grow with Optomatica's needs while maintaining the Zen-inspired, productivity-focused vision outlined in the Product Design Document.

### Next Steps

1. **Set up development environment** with Docker and monorepo structure
2. **Implement core authentication** with Optomatica domain verification
3. **Build minimum viable timer** with offline capabilities
4. **Add basic task management** with AI-powered suggestions
5. **Implement team features** and real-time collaboration
6. **Add gamification** and progress tracking
7. **Integrate third-party services** (calendars, communication tools)
8. **Deploy to production** with comprehensive monitoring

This technical foundation will enable the development team to build a robust, scalable, and delightful productivity application that transforms how Optomatica employees approach focused work and team collaboration.