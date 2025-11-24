# Comprehensive Testing Strategy

## 🎯 Testing Overview

Complete testing strategy for backend analytics fixes and dashboard implementation, ensuring production-ready quality and reliability.

### Coverage Goals
- **Unit Tests**: 90%+ code coverage
- **Integration Tests**: 100% API endpoint coverage
- **E2E Tests**: Critical user journey coverage
- **Performance Tests**: Load and stress testing
- **Accessibility Tests**: WCAG 2.1 AA compliance

---

## 🧪 Backend Testing Strategy

### 1. Unit Tests - Analytics Service

#### Test Structure
```typescript
// packages/backend/src/analytics/__tests__/analytics.service.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { AnalyticsService } from '../analytics.service';
import { DatabaseService } from '../../config/database.config';
import { NotFoundException, ForbiddenException } from '@nestjs/common';

describe('AnalyticsService', () => {
  let service: AnalyticsService;
  let prisma: jest.Mocked<DatabaseService>;

  beforeEach(async () => {
    const mockPrisma = {
      session: { findMany: jest.fn(), count: jest.fn() },
      task: { count: jest.fn(), findMany: jest.fn() },
      user: { findUnique: jest.fn() },
      team: { findUnique: jest.fn() },
      teamMember: { findUnique: jest.fn() },
    } as any;

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AnalyticsService,
        {
          provide: DatabaseService,
          useValue: mockPrisma,
        },
      ],
    }).compile();

    service = module.get<AnalyticsService>(AnalyticsService);
    prisma = module.get<DatabaseService>(DatabaseService);
  });

  describe('getTeamMemberCompletionRate', () => {
    const mockUserId = 'user-123';
    const mockStartDate = new Date('2024-01-01');
    const mockEndDate = new Date('2024-01-31');

    it('should calculate 0% for user with no tasks', async () => {
      prisma.task.count
        .mockResolvedValueOnce(0) // total tasks
        .mockResolvedValueOnce(0); // completed tasks

      const result = await service.getTeamMemberCompletionRate(
        mockUserId,
        mockStartDate,
        mockEndDate
      );

      expect(result).toBe(0);
      expect(prisma.task.count).toHaveBeenCalledTimes(2);
    });

    it('should calculate 100% for user with all tasks completed', async () => {
      prisma.task.count
        .mockResolvedValueOnce(10) // total tasks
        .mockResolvedValueOnce(10); // completed tasks

      const result = await service.getTeamMemberCompletionRate(
        mockUserId,
        mockStartDate,
        mockEndDate
      );

      expect(result).toBe(100);
    });

    it('should calculate correct percentage for mixed task status', async () => {
      prisma.task.count
        .mockResolvedValueOnce(10) // total tasks
        .mockResolvedValueOnce(7); // completed tasks

      const result = await service.getTeamMemberCompletionRate(
        mockUserId,
        mockStartDate,
        mockEndDate
      );

      expect(result).toBe(70);
    });

    it('should respect date range filtering', async () => {
      prisma.task.count
        .mockResolvedValueOnce(5) // total tasks in date range
        .mockResolvedValueOnce(3); // completed tasks in date range

      const result = await service.getTeamMemberCompletionRate(
        mockUserId,
        mockStartDate,
        mockEndDate
      );

      expect(result).toBe(60);
      expect(prisma.task.count).toHaveBeenCalledWith({
        where: {
          OR: [
            { creatorId: mockUserId },
            { assigneeId: mockUserId }
          ],
          createdAt: {
            gte: mockStartDate,
            lte: mockEndDate
          }
        }
      });
    });

    it('should handle edge case with null date range', async () => {
      prisma.task.count
        .mockResolvedValueOnce(8)
        .mockResolvedValueOnce(6);

      const result = await service.getTeamMemberCompletionRate(mockUserId);

      expect(result).toBe(75);
      expect(prisma.task.count).toHaveBeenCalledWith({
        where: {
          OR: [
            { creatorId: mockUserId },
            { assigneeId: mockUserId }
          ]
        }
      });
    });
  });

  describe('getFocusAnalytics', () => {
    const mockUserId = 'user-123';
    const mockSessions = [
      { duration: 25, startTime: new Date('2024-01-15T09:00:00Z'), quality: 5 },
      { duration: 25, startTime: new Date('2024-01-15T10:00:00Z'), quality: 4 },
      { duration: 5, startTime: new Date('2024-01-15T10:30:00Z'), quality: null, type: 'SHORT_BREAK' },
    ];

    it('should calculate focus analytics correctly', async () => {
      prisma.session.findMany.mockResolvedValue(mockSessions);

      const result = await service.getFocusAnalytics(mockUserId);

      expect(result.dailyFocusTime).toBe(50); // 2 * 25 min sessions today
      expect(result.weeklyFocusTime).toBeGreaterThan(0);
      expect(result.monthlyFocusTime).toBeGreaterThan(0);
      expect(result.averageSessionLength).toBe(25);
      expect(result.completionRate).toBeGreaterThan(0);
      expect(result.focusTrend).toBeDefined();
    });

    it('should handle user with no sessions', async () => {
      prisma.session.findMany.mockResolvedValue([]);

      const result = await service.getFocusAnalytics(mockUserId);

      expect(result.dailyFocusTime).toBe(0);
      expect(result.weeklyFocusTime).toBe(0);
      expect(result.monthlyFocusTime).toBe(0);
      expect(result.averageSessionLength).toBe(0);
      expect(result.completionRate).toBe(0);
    });

    it('should respect date range filtering', async () => {
      const startDate = new Date('2024-01-01T00:00:00Z');
      const endDate = new Date('2024-01-31T23:59:59Z');

      prisma.session.findMany.mockResolvedValue(mockSessions);

      await service.getFocusAnalytics(mockUserId, startDate, endDate);

      expect(prisma.session.findMany).toHaveBeenCalledWith({
        where: {
          userId: mockUserId,
          completed: true,
          type: 'POMODORO',
          startTime: {
            gte: startDate,
            lte: endDate
          }
        },
        select: {
          duration: true,
          startTime: true,
          quality: true,
          taskId: true
        },
        orderBy: { startTime: 'desc' }
      });
    });
  });

  describe('getWellnessAnalytics', () => {
    const mockUserId = 'user-123';
    const mockUser = {
      wellnessScore: 85,
      streak: 10,
      totalFocusTime: 1500, // 25 hours
      preferences: JSON.stringify({
        wellness: {
          hydrationGoal: 8,
          movementGoal: 5
        }
      })
    };

    it('should calculate wellness analytics from user data', async () => {
      prisma.user.findUnique.mockResolvedValue(mockUser);

      const result = await service.getWellnessAnalytics(mockUserId);

      expect(result.mindfulnessMinutes).toBe(150); // 10% of focus time
      expect(result.hydrationGoal).toBe(8);
      expect(result.movementGoal).toBe(5);
      expect(result.moodRating).toBe(85); // From wellness score
      expect(result.stressLevel).toBeLessThanOrEqual(5);
      expect(result.energyLevel).toBeGreaterThan(0);
    });

    it('should handle user with no wellness score', async () => {
      const userWithoutWellness = {
        ...mockUser,
        wellnessScore: null
      };
      prisma.user.findUnique.mockResolvedValue(userWithoutWellness);

      const result = await service.getWellnessAnalytics(mockUserId);

      expect(result.moodRating).toBe(3); // Default mood
      expect(result.stressLevel).toBe(3); // Default stress
      expect(result.energyLevel).toBeGreaterThan(0);
    });

    it('should throw NotFoundException for non-existent user', async () => {
      prisma.user.findUnique.mockResolvedValue(null);

      await expect(service.getWellnessAnalytics('invalid-user'))
        .rejects.toThrow(NotFoundException);
    });
  });

  describe('getTeamAnalytics', () => {
    const mockTeamId = 'team-123';
    const mockUserId = 'user-123';
    const mockTeam = {
      id: mockTeamId,
      name: 'Development Team',
      members: [
        {
          userId: 'user-1',
          user: {
            id: 'user-1',
            firstName: 'John',
            lastName: 'Doe',
            wellnessScore: 90,
            level: 5,
            xp: 1500,
            streak: 15
          }
        },
        {
          userId: 'user-2',
          user: {
            id: 'user-2',
            firstName: 'Jane',
            lastName: 'Smith',
            wellnessScore: 80,
            level: 4,
            xp: 800,
            streak: 8
          }
        }
      ]
    };

    it('should calculate team analytics correctly', async () => {
      prisma.teamMember.findUnique.mockResolvedValue({ userId: mockUserId });
      prisma.team.findUnique.mockResolvedValue(mockTeam);

      const mockSessions = [
        { userId: 'user-1', duration: 120, quality: 5, startTime: new Date() },
        { userId: 'user-2', duration: 90, quality: 4, startTime: new Date() }
      ];
      prisma.session.findMany.mockResolvedValue(mockSessions);

      // Mock task completion counts
      prisma.task.count
        .mockResolvedValueOnce(5) // user-1 total tasks
        .mockResolvedValueOnce(4) // user-1 completed tasks
        .mockResolvedValueOnce(3) // user-2 total tasks
        .mockResolvedValueOnce(2); // user-2 completed tasks

      const result = await service.getTeamAnalytics(mockTeamId, undefined, undefined, mockUserId);

      expect(result.teamId).toBe(mockTeamId);
      expect(result.teamName).toBe('Development Team');
      expect(result.memberCount).toBe(2);
      expect(result.totalFocusTime).toBe(210);
      expect(result.averageFocusTime).toBe(105);
      expect(result.topPerformers).toHaveLength(2);
      expect(result.focusTrend).toBeDefined();
      expect(result.wellnessScore).toBe(85); // Average of 90 and 80
    });

    it('should throw ForbiddenException for non-team member', async () => {
      prisma.teamMember.findUnique.mockResolvedValue(null);

      await expect(service.getTeamAnalytics(mockTeamId, undefined, undefined, 'non-member'))
        .rejects.toThrow(ForbiddenException);
    });

    it('should throw NotFoundException for non-existent team', async () => {
      prisma.teamMember.findUnique.mockResolvedValue({ userId: mockUserId });
      prisma.team.findUnique.mockResolvedValue(null);

      await expect(service.getTeamAnalytics('invalid-team', undefined, undefined, mockUserId))
        .rejects.toThrow(NotFoundException);
    });
  });
});
```

### 2. Integration Tests - API Endpoints

#### Analytics Controller Tests
```typescript
// packages/backend/src/analytics/__tests__/analytics.controller.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { AnalyticsController } from '../analytics.controller';
import { AnalyticsService } from '../analytics.service';
import { JwtAuthGuard } from '../../auth/guards/jwt-auth.guard';

describe('AnalyticsController', () => {
  let controller: AnalyticsController;
  let analyticsService: jest.Mocked<AnalyticsService>;

  beforeEach(async () => {
    const mockAnalyticsService = {
      getFocusAnalytics: jest.fn(),
      getWellnessAnalytics: jest.fn(),
      getTeamAnalytics: jest.fn(),
    } as any;

    const module: TestingModule = await Test.createTestingModule({
      controllers: [AnalyticsController],
      providers: [
        {
          provide: AnalyticsService,
          useValue: mockAnalyticsService,
        },
      ],
    })
    .overrideGuard(JwtAuthGuard)
    .useValue({ canActivate: () => true })
    .compile();

    controller = module.get<AnalyticsController>(AnalyticsController);
    analyticsService = module.get<AnalyticsService>(AnalyticsService);
  });

  describe('getFocusAnalytics', () => {
    it('should return focus analytics for authenticated user', async () => {
      const mockUser = { id: 'user-123', email: 'test@example.com' };
      const mockAnalytics = {
        dailyFocusTime: 225,
        weeklyFocusTime: 1575,
        monthlyFocusTime: 6750,
        averageSessionLength: 25.5,
        peakFocusHours: [9, 10, 14],
        focusTrend: 'IMPROVING',
        completionRate: 85.5
      };

      analyticsService.getFocusAnalytics.mockResolvedValue(mockAnalytics);

      const result = await controller.getFocusAnalytics(
        { user: mockUser } as any,
        { startDate: '2024-01-01T00:00:00.000Z', endDate: '2024-01-31T23:59:59.999Z' }
      );

      expect(result).toEqual(mockAnalytics);
      expect(analyticsService.getFocusAnalytics).toHaveBeenCalledWith(
        'user-123',
        new Date('2024-01-01T00:00:00.000Z'),
        new Date('2024-01-31T23:59:59.999Z')
      );
    });

    it('should handle date range parameters', async () => {
      const mockUser = { id: 'user-123' };
      const mockAnalytics = { dailyFocusTime: 150 };

      analyticsService.getFocusAnalytics.mockResolvedValue(mockAnalytics);

      await controller.getFocusAnalytics(
        { user: mockUser } as any,
        { startDate: '2024-01-15T00:00:00.000Z' }
      );

      expect(analyticsService.getFocusAnalytics).toHaveBeenCalledWith(
        'user-123',
        new Date('2024-01-15T00:00:00.000Z'),
        undefined
      );
    });

    it('should handle no date parameters', async () => {
      const mockUser = { id: 'user-123' };
      const mockAnalytics = { dailyFocusTime: 200 };

      analyticsService.getFocusAnalytics.mockResolvedValue(mockAnalytics);

      await controller.getFocusAnalytics({ user: mockUser } as any, {});

      expect(analyticsService.getFocusAnalytics).toHaveBeenCalledWith(
        'user-123',
        undefined,
        undefined
      );
    });
  });

  describe('getWellnessAnalytics', () => {
    it('should return wellness analytics for authenticated user', async () => {
      const mockUser = { id: 'user-123' };
      const mockWellness = {
        mindfulnessMinutes: 45,
        hydrationGoal: 8,
        hydrationCurrent: 6,
        movementGoal: 5,
        movementCurrent: 3,
        moodRating: 4,
        stressLevel: 2,
        energyLevel: 4
      };

      analyticsService.getWellnessAnalytics.mockResolvedValue(mockWellness);

      const result = await controller.getWellnessAnalytics(
        { user: mockUser } as any,
        {}
      );

      expect(result).toEqual(mockWellness);
      expect(analyticsService.getWellnessAnalytics).toHaveBeenCalledWith(
        'user-123',
        undefined,
        undefined
      );
    });
  });

  describe('getTeamAnalytics', () => {
    it('should return team analytics for team member', async () => {
      const mockUser = { id: 'user-123' };
      const mockTeamAnalytics = {
        teamId: 'team-123',
        teamName: 'Dev Team',
        memberCount: 5,
        totalFocusTime: 3600,
        averageFocusTime: 720,
        tasksCompleted: 25,
        averageCompletionRate: 85,
        topPerformers: [],
        focusTrend: 'IMPROVING',
        wellnessScore: 82,
        collaborationScore: 75,
        period: {
          startDate: '2024-01-01T00:00:00.000Z',
          endDate: '2024-01-31T23:59:59.999Z'
        }
      };

      analyticsService.getTeamAnalytics.mockResolvedValue(mockTeamAnalytics);

      const result = await controller.getTeamAnalytics(
        'team-123',
        { user: mockUser } as any,
        { startDate: '2024-01-01T00:00:00.000Z', endDate: '2024-01-31T23:59:59.999Z' }
      );

      expect(result).toEqual(mockTeamAnalytics);
      expect(analyticsService.getTeamAnalytics).toHaveBeenCalledWith(
        'team-123',
        new Date('2024-01-01T00:00:00.000Z'),
        new Date('2024-01-31T23:59:59.999Z'),
        'user-123'
      );
    });
  });
});
```

### 3. API Endpoint Testing with Supertest

```typescript
// packages/backend/src/analytics/__tests__/analytics.e2e.spec.ts
import request from 'supertest';
import { INestApplication } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import { AppModule } from '../../app.module';

describe('Analytics API (e2e)', () => {
  let app: INestApplication;
  let authToken: string;

  beforeAll(async () => {
    const moduleFixture = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    await app.init();

    // Get auth token
    const loginResponse = await request(app.getHttpServer())
      .post('/api/auth/login')
      .send({
        email: 'test@optomatica.com',
        password: 'password123'
      });

    authToken = loginResponse.body.token;
  });

  afterAll(async () => {
    await app.close();
  });

  describe('GET /api/analytics/focus', () => {
    it('should return focus analytics for authenticated user', async () => {
      const response = await request(app.getHttpServer())
        .get('/api/analytics/focus')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toMatchObject({
        dailyFocusTime: expect.any(Number),
        weeklyFocusTime: expect.any(Number),
        monthlyFocusTime: expect.any(Number),
        averageSessionLength: expect.any(Number),
        peakFocusHours: expect.any(Array),
        focusTrend: expect.stringMatching(/IMPROVING|DECLINING|STABLE/),
        completionRate: expect.any(Number)
      });

      expect(response.body.dailyFocusTime).toBeGreaterThanOrEqual(0);
      expect(response.body.completionRate).toBeLessThanOrEqual(100);
    });

    it('should accept date range parameters', async () => {
      const response = await request(app.getHttpServer())
        .get('/api/analytics/focus')
        .query({
          startDate: '2024-01-01T00:00:00.000Z',
          endDate: '2024-01-31T23:59:59.999Z'
        })
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toBeDefined();
    });

    it('should reject unauthenticated requests', async () => {
      await request(app.getHttpServer())
        .get('/api/analytics/focus')
        .expect(401);
    });

    it('should validate date format', async () => {
      await request(app.getHttpServer())
        .get('/api/analytics/focus')
        .query({ startDate: 'invalid-date' })
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400);
    });
  });

  describe('GET /api/analytics/wellness', () => {
    it('should return wellness analytics for authenticated user', async () => {
      const response = await request(app.getHttpServer())
        .get('/api/analytics/wellness')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toMatchObject({
        mindfulnessMinutes: expect.any(Number),
        hydrationGoal: expect.any(Number),
        hydrationCurrent: expect.any(Number),
        movementGoal: expect.any(Number),
        movementCurrent: expect.any(Number),
        moodRating: expect.any(Number),
        stressLevel: expect.any(Number),
        energyLevel: expect.any(Number)
      });

      expect(response.body.moodRating).toBeBetween(1, 5);
      expect(response.body.stressLevel).toBeBetween(1, 5);
      expect(response.body.energyLevel).toBeBetween(1, 5);
    });
  });

  describe('GET /api/analytics/teams/:teamId', () => {
    const teamId = 'test-team-123';

    it('should return team analytics for team member', async () => {
      // First, ensure user is part of team (setup in test database)
      const response = await request(app.getHttpServer())
        .get(`/api/analytics/teams/${teamId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toMatchObject({
        teamId,
        teamName: expect.any(String),
        memberCount: expect.any(Number),
        totalFocusTime: expect.any(Number),
        averageFocusTime: expect.any(Number),
        tasksCompleted: expect.any(Number),
        averageCompletionRate: expect.any(Number),
        topPerformers: expect.any(Array),
        focusTrend: expect.stringMatching(/IMPROVING|DECLINING|STABLE/),
        wellnessScore: expect.any(Number),
        collaborationScore: expect.any(Number),
        period: expect.objectContaining({
          startDate: expect.any(String),
          endDate: expect.any(String)
        })
      });
    });

    it('should reject non-team members', async () => {
      const nonMemberTeamId = 'other-team-456';

      await request(app.getHttpServer())
        .get(`/api/analytics/teams/${nonMemberTeamId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(403);
    });

    it('should handle non-existent team', async () => {
      await request(app.getHttpServer())
        .get('/api/analytics/teams/non-existent-team')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);
    });
  });
});
```

---

## 🎨 Frontend Testing Strategy

### 1. Component Unit Tests

#### Dashboard Screen Component Tests
```typescript
// packages/frontend/src/components/pages/DashboardScreen/DashboardScreen.test.tsx
import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { configureStore } from '@reduxjs/toolkit';
import { DashboardScreen } from './DashboardScreen';
import { apiSlice } from '../../../store/api/apiSlice';

// Mock RTK Query hooks
jest.mock('../../../store/api/apiSlice', () => ({
  apiSlice: {
    useGetFocusAnalyticsQuery: jest.fn(),
    useGetWellnessAnalyticsQuery: jest.fn(),
  },
}));

const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      api: apiSlice.reducer,
      // Add other reducers as needed
    },
    preloadedState: initialState,
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware().concat(apiSlice.middleware),
  });
};

const mockFocusAnalytics = {
  dailyFocusTime: 225,
  weeklyFocusTime: 1575,
  monthlyFocusTime: 6750,
  averageSessionLength: 25.5,
  peakFocusHours: [9, 10, 14],
  focusTrend: 'IMPROVING' as const,
  completionRate: 85.5,
};

const mockWellnessAnalytics = {
  mindfulnessMinutes: 45,
  hydrationGoal: 8,
  hydrationCurrent: 6,
  movementGoal: 5,
  movementCurrent: 3,
  moodRating: 4,
  stressLevel: 2,
  energyLevel: 4,
};

describe('DashboardScreen', () => {
  let mockStore: ReturnType<typeof createMockStore>;

  beforeEach(() => {
    mockStore = createMockStore({
      auth: {
        user: { id: 'user-123', firstName: 'John', lastName: 'Doe' },
        token: 'mock-token',
        isAuthenticated: true,
      },
    });

    jest.clearAllMocks();
  });

  const renderDashboard = () => {
    return render(
      <Provider store={mockStore}>
        <BrowserRouter>
          <DashboardScreen />
        </BrowserRouter>
      </Provider>
    );
  };

  it('should render dashboard loading state', () => {
    const { useGetFocusAnalyticsQuery, useGetWellnessAnalyticsQuery } = require('../../../store/api/apiSlice');

    useGetFocusAnalyticsQuery.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    });
    useGetWellnessAnalyticsQuery.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    });

    renderDashboard();

    expect(screen.getByText('Loading analytics...')).toBeInTheDocument();
  });

  it('should render dashboard with analytics data', async () => {
    const { useGetFocusAnalyticsQuery, useGetWellnessAnalyticsQuery } = require('../../../store/api/apiSlice');

    useGetFocusAnalyticsQuery.mockReturnValue({
      data: mockFocusAnalytics,
      isLoading: false,
      error: null,
    });
    useGetWellnessAnalyticsQuery.mockReturnValue({
      data: mockWellnessAnalytics,
      isLoading: false,
      error: null,
    });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Today\'s Focus')).toBeInTheDocument();
      expect(screen.getByText('Wellness Metrics')).toBeInTheDocument();
      expect(screen.getByText('3h 45m')).toBeInTheDocument(); // 225 minutes formatted
      expect(screen.getByText('85%')).toBeInTheDocument(); // completion rate
    });
  });

  it('should display error state on API failure', async () => {
    const { useGetFocusAnalyticsQuery, useGetWellnessAnalyticsQuery } = require('../../../store/api/apiSlice');

    useGetFocusAnalyticsQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: { message: 'Failed to load analytics' },
    });
    useGetWellnessAnalyticsQuery.mockReturnValue({
      data: mockWellnessAnalytics,
      isLoading: false,
      error: null,
    });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Failed to load focus analytics')).toBeInTheDocument();
    });
  });

  it('should handle date range selection', async () => {
    const { useGetFocusAnalyticsQuery, useGetWellnessAnalyticsQuery } = require('../../../store/api/apiSlice');

    useGetFocusAnalyticsQuery.mockReturnValue({
      data: mockFocusAnalytics,
      isLoading: false,
      error: null,
    });
    useGetWellnessAnalyticsQuery.mockReturnValue({
      data: mockWellnessAnalytics,
      isLoading: false,
      error: null,
    });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Today\'s Focus')).toBeInTheDocument();
    });

    // Test date range picker
    const dateRangeButton = screen.getByLabelText('Select date range');
    fireEvent.click(dateRangeButton);

    const thisWeekOption = screen.getByText('This Week');
    fireEvent.click(thisWeekOption);

    // Verify API was called with new date range
    expect(useGetFocusAnalyticsQuery).toHaveBeenCalledWith({
      startDate: expect.any(String),
      endDate: expect.any(String),
    });
  });

  it('should format time correctly', async () => {
    const { useGetFocusAnalyticsQuery, useGetWellnessAnalyticsQuery } = require('../../../store/api/apiSlice');

    const testFocusData = {
      ...mockFocusAnalytics,
      dailyFocusTime: 150, // 2.5 hours
      weeklyFocusTime: 900, // 15 hours
    };

    useGetFocusAnalyticsQuery.mockReturnValue({
      data: testFocusData,
      isLoading: false,
      error: null,
    });
    useGetWellnessAnalyticsQuery.mockReturnValue({
      data: mockWellnessAnalytics,
      isLoading: false,
      error: null,
    });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('2h 30m')).toBeInTheDocument(); // daily
      expect(screen.getByText('15h')).toBeInTheDocument(); // weekly
    });
  });

  it('should be accessible', async () => {
    const { useGetFocusAnalyticsQuery, useGetWellnessAnalyticsQuery } = require('../../../store/api/apiSlice');

    useGetFocusAnalyticsQuery.mockReturnValue({
      data: mockFocusAnalytics,
      isLoading: false,
      error: null,
    });
    useGetWellnessAnalyticsQuery.mockReturnValue({
      data: mockWellnessAnalytics,
      isLoading: false,
      error: null,
    });

    renderDashboard();

    await waitFor(() => {
      const mainHeading = screen.getByRole('heading', { name: 'Progress Dashboard' });
      expect(mainHeading).toBeInTheDocument();

      // Check for ARIA labels on progress indicators
      const progressBar = screen.getByRole('progressbar');
      expect(progressBar).toHaveAttribute('aria-label');
      expect(progressBar).toHaveAttribute('aria-valuenow');
    });
  });
});
```

#### Focus Metrics Card Component Tests
```typescript
// packages/frontend/src/components/organisms/FocusAnalytics/FocusMetricsCard.test.tsx
import React from 'react';
import { render, screen } from '@testing-library/react';
import { FocusMetricsCard } from './FocusMetricsCard';
import { theme } from '../../../theme';

describe('FocusMetricsCard', () => {
  const defaultProps = {
    dailyFocusTime: 225,
    weeklyFocusTime: 1575,
    monthlyFocusTime: 6750,
    averageSessionLength: 25.5,
    completionRate: 85.5,
    focusTrend: 'IMPROVING' as const,
    streak: 7,
    dailyGoal: 300,
  };

  it('should render focus metrics correctly', () => {
    render(<FocusMetricsCard {...defaultProps} />);

    expect(screen.getByText('Today\'s Focus')).toBeInTheDocument();
    expect(screen.getByText('3h 45m')).toBeInTheDocument(); // 225 minutes
    expect(screen.getByText('75%')).toBeInTheDocument(); // progress towards goal
    expect(screen.getByText('🔥 7 day streak')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument(); // completion rate
  });

  it('should show correct progress for daily goal', () => {
    render(<FocusMetricsCard {...defaultProps} />);

    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toHaveAttribute('aria-valuenow', '75'); // 225/300 = 75%
  });

  it('should display trend indicator correctly', () => {
    render(<FocusMetricsCard {...defaultProps} focusTrend="DECLINING" />);

    expect(screen.getByText('Declining vs last week')).toBeInTheDocument();
    expect(screen.getByTestId('trend-indicator')).toHaveClass('trend-declining');
  });

  it('should handle zero values gracefully', () => {
    const zeroProps = {
      ...defaultProps,
      dailyFocusTime: 0,
      weeklyFocusTime: 0,
      streak: 0,
    };

    render(<FocusMetricsCard {...zeroProps} />);

    expect(screen.getByText('0m')).toBeInTheDocument();
    expect(screen.getByText('No active streak')).toBeInTheDocument();
  });

  it('should format large time values correctly', () => {
    const largeProps = {
      ...defaultProps,
      monthlyFocusTime: 3000, // 50 hours
    };

    render(<FocusMetricsCard {...largeProps} />);

    expect(screen.getByText('50h')).toBeInTheDocument();
  });

  it('should be accessible', () => {
    render(<FocusMetricsCard {...defaultProps} />);

    // Check for proper ARIA labels
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    expect(screen.getByLabelText('Daily focus progress')).toBeInTheDocument();

    // Check keyboard navigation
    const focusableElements = screen.getAllByRole('button');
    focusableElements.forEach(element => {
      expect(element).toHaveAttribute('tabindex');
    });
  });
});
```

### 2. Integration Tests - API Integration

#### Dashboard Data Integration Tests
```typescript
// packages/frontend/src/components/pages/DashboardScreen/DashboardScreen.integration.test.tsx
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import { configureStore } from '@reduxjs/toolkit';
import { DashboardScreen } from './DashboardScreen';
import { apiSlice } from '../../../store/api/apiSlice';

// Mock server setup
const server = setupServer(
  rest.get('/api/analytics/focus', (req, res, ctx) => {
    return res(ctx.json({
      dailyFocusTime: 225,
      weeklyFocusTime: 1575,
      monthlyFocusTime: 6750,
      averageSessionLength: 25.5,
      peakFocusHours: [9, 10, 14],
      focusTrend: 'IMPROVING',
      completionRate: 85.5,
    }));
  }),
  rest.get('/api/analytics/wellness', (req, res, ctx) => {
    return res(ctx.json({
      mindfulnessMinutes: 45,
      hydrationGoal: 8,
      hydrationCurrent: 6,
      movementGoal: 5,
      movementCurrent: 3,
      moodRating: 4,
      stressLevel: 2,
      energyLevel: 4,
    }));
  }),
  rest.get('/api/analytics/focus', (req, res, ctx) => {
    return res(ctx.status(500), ctx.json({ message: 'Server error' }));
  }),
);

describe('DashboardScreen Integration', () => {
  let store: ReturnType<typeof configureStore>;

  beforeAll(() => server.listen());
  afterEach(() => server.resetHandlers());
  afterAll(() => server.close());

  beforeEach(() => {
    store = configureStore({
      reducer: {
        api: apiSlice.reducer,
      },
      middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware().concat(apiSlice.middleware),
    });
  });

  const renderDashboard = () => {
    return render(
      <Provider store={store}>
        <BrowserRouter>
          <DashboardScreen />
        </BrowserRouter>
      </Provider>
    );
  };

  it('should load and display analytics data', async () => {
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Today\'s Focus')).toBeInTheDocument();
      expect(screen.getByText('3h 45m')).toBeInTheDocument();
      expect(screen.getByText('Wellness Metrics')).toBeInTheDocument();
      expect(screen.getByText('6/8 glasses')).toBeInTheDocument();
    });
  });

  it('should handle API errors gracefully', async () => {
    server.use(
      rest.get('/api/analytics/focus', (req, res, ctx) => {
        return res(ctx.status(500), ctx.json({ message: 'Server error' }));
      })
    );

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/Failed to load analytics/)).toBeInTheDocument();
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });
  });

  it('should refetch data on retry', async () => {
    server.use(
      rest.get('/api/analytics/focus', (req, res, ctx) => {
        return res(ctx.status(500));
      })
    );

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });

    // Fix the server and retry
    server.resetHandlers();
    fireEvent.click(screen.getByText('Retry'));

    await waitFor(() => {
      expect(screen.getByText('3h 45m')).toBeInTheDocument();
    });
  });

  it('should respect date range parameters', async () => {
    renderDashboard();

    // Find and click date range selector
    const dateRangeSelector = screen.getByLabelText('Select date range');
    fireEvent.click(dateRangeSelector);

    const thisWeekOption = screen.getByText('This Week');
    fireEvent.click(thisWeekOption);

    await waitFor(() => {
      expect(screen.getByText('This Week')).toBeInTheDocument();
    });

    // Verify correct API call was made
    expect(screen.getByText('Loading analytics...')).toBeInTheDocument();
  });
});
```

### 3. E2E Tests with Cypress

```typescript
// cypress/e2e/dashboard.cy.ts
describe('Dashboard E2E', () => {
  beforeEach(() => {
    // Login and navigate to dashboard
    cy.login('test@optomatica.com', 'password123');
    cy.visit('/dashboard');
  });

  it('should display dashboard with real data', () => {
    cy.contains('Progress Dashboard').should('be.visible');
    cy.contains('Today\'s Focus').should('be.visible');
    cy.contains('Wellness Metrics').should('be.visible');

    // Wait for data to load
    cy.get('[data-testid="loading-spinner"]').should('not.exist');

    // Verify focus metrics
    cy.get('[data-testid="daily-focus-time"]').should('contain.text', 'h');
    cy.get('[data-testid="completion-rate"]').should('contain.text', '%');

    // Verify wellness metrics
    cy.get('[data-testid="hydration-progress"]').should('be.visible');
    cy.get('[data-testid="movement-count"]').should('be.visible');
  });

  it('should allow date range selection', () => {
    cy.get('[data-testid="date-range-selector"]').click();
    cy.get('[data-testid="option-this-week"]').click();

    // Should trigger data refresh
    cy.get('[data-testid="loading-spinner"]').should('be.visible');
    cy.get('[data-testid="loading-spinner"]').should('not.exist');

    // URL should reflect date range
    cy.url().should('include', 'startDate=');
    cy.url().should('include', 'endDate=');
  });

  it('should display weekly chart correctly', () => {
    cy.get('[data-testid="weekly-chart"]').should('be.visible');

    // Check for chart elements
    cy.get('[data-testid="chart-bar"]').should('have.length.greaterThan', 0);
    cy.get('[data-testid="chart-tooltip"]').should('exist');
  });

  it('should show achievement badges', () => {
    cy.get('[data-testid="achievement-gallery"]').should('be.visible');
    cy.get('[data-testid="achievement-badge"]').should('have.length.greaterThan', 0);
  });

  it('should be responsive on mobile', () => {
    cy.viewport('iphone-x');

    // Mobile layout should be single column
    cy.get('[data-testid="dashboard-grid"]').should('have.css', 'grid-template-columns', '1fr');

    // Navigation should be bottom bar
    cy.get('[data-testid="mobile-nav"]').should('be.visible');
  });

  it('should be responsive on tablet', () => {
    cy.viewport('ipad-2');

    // Tablet layout should be two columns
    cy.get('[data-testid="dashboard-grid"]').should('have.css', 'grid-template-columns', 'repeat(2, 1fr)');
  });

  it('should be responsive on desktop', () => {
    cy.viewport(1280, 720);

    // Desktop layout should be three columns
    cy.get('[data-testid="dashboard-grid"]').should('have.css', 'grid-template-columns', 'repeat(3, 1fr)');
  });

  it('should handle network errors gracefully', () => {
    // Simulate network failure
    cy.intercept('GET', '/api/analytics/focus', { forceNetworkError: true });
    cy.intercept('GET', '/api/analytics/wellness', { forceNetworkError: true });

    cy.reload();

    cy.get('[data-testid="error-message"]').should('be.visible');
    cy.get('[data-testid="retry-button"]').should('be.visible');

    // Test retry functionality
    cy.intercept('GET', '/api/analytics/focus', { fixture: 'focus-analytics.json' });
    cy.intercept('GET', '/api/analytics/wellness', { fixture: 'wellness-analytics.json' });

    cy.get('[data-testid="retry-button"]').click();

    cy.get('[data-testid="error-message"]').should('not.exist');
    cy.get('[data-testid="daily-focus-time"]').should('be.visible');
  });

  it('should support keyboard navigation', () => {
    // Tab through all interactive elements
    cy.focused().tab();

    // Should focus on first interactive element
    cy.get('[data-testid="date-range-selector"]').should('be.focused');

    // Continue tabbing through elements
    cy.focused().tab();
    cy.focused().tab();

    // All interactive elements should be focusable
    cy.get('[tabindex="0"]').should('have.length.greaterThan', 0);
  });

  it('should meet accessibility standards', () => {
    // Run accessibility audit
    cy.injectAxe();
    cy.checkA11y();

    // Test screen reader compatibility
    cy.get('[role="progressbar"]').should('have.attr', 'aria-label');
    cy.get('[role="button"]').should('have.attr', 'aria-label');

    // Test color contrast (via axe)
    cy.checkA11y({ includedImpacts: ['color-contrast'] });
  });

  it('should have good performance', () => {
    // Measure load time
    cy.window().then((win) => {
      const performanceData = win.performance.getEntriesByType('navigation')[0];
      expect(performanceData.loadEventEnd - performanceData.navigationStart).to.be.lessThan(3000);
    });

    // Test animation performance
    cy.get('[data-testid="progress-ring"]').should('have.css', 'transition');
    cy.get('[data-testid="chart-animation"]').should('have.css', 'animation-duration');
  });
});
```

---

## ⚡ Performance Testing

### 1. Load Testing - Backend API

```typescript
// packages/backend/src/analytics/__tests__/performance.test.ts
import request from 'supertest';
import { INestApplication } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import { AppModule } from '../../app.module';

describe('Analytics Performance Tests', () => {
  let app: INestApplication;
  let authToken: string;

  beforeAll(async () => {
    const moduleFixture = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    await app.init();

    // Create test user and get token
    const loginResponse = await request(app.getHttpServer())
      .post('/api/auth/login')
      .send({ email: 'perf-test@optomatica.com', password: 'password123' });

    authToken = loginResponse.body.token;

    // Create test data for performance testing
    await createTestData();
  });

  describe('Focus Analytics Performance', () => {
    it('should handle concurrent requests', async () => {
      const concurrentRequests = 50;
      const startTime = Date.now();

      const promises = Array.from({ length: concurrentRequests }, () =>
        request(app.getHttpServer())
          .get('/api/analytics/focus')
          .set('Authorization', `Bearer ${authToken}`)
          .expect(200)
      );

      await Promise.all(promises);
      const endTime = Date.now();

      const totalTime = endTime - startTime;
      const averageTime = totalTime / concurrentRequests;

      // Each request should average under 100ms
      expect(averageTime).toBeLessThan(100);

      // Total time should be under 5 seconds for 50 requests
      expect(totalTime).toBeLessThan(5000);
    });

    it('should handle large date ranges efficiently', async () => {
      const startTime = Date.now();

      await request(app.getHttpServer())
        .get('/api/analytics/focus')
        .query({
          startDate: '2023-01-01T00:00:00.000Z',
          endDate: '2024-01-31T23:59:59.999Z' // 13 months of data
        })
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      const responseTime = Date.now() - startTime;

      // Large date range queries should be under 500ms
      expect(responseTime).toBeLessThan(500);
    });

    it('should maintain performance under load', async () => {
      const duration = 10000; // 10 seconds
      const startTime = Date.now();
      let requestCount = 0;

      while (Date.now() - startTime < duration) {
        await request(app.getHttpServer())
          .get('/api/analytics/focus')
          .set('Authorization', `Bearer ${authToken}`)
          .expect(200);
        requestCount++;
      }

      const requestsPerSecond = requestCount / (duration / 1000);

      // Should handle at least 10 requests per second
      expect(requestsPerSecond).toBeGreaterThan(10);
    });
  });

  describe('Team Analytics Performance', () => {
    it('should handle teams with many members', async () => {
      // Create large team for testing
      const largeTeamId = await createLargeTeam(100); // 100 members
      const memberToken = await addMemberToTeam(largeTeamId);

      const startTime = Date.now();

      await request(app.getHttpServer())
        .get(`/api/analytics/teams/${largeTeamId}`)
        .set('Authorization', `Bearer ${memberToken}`)
        .expect(200);

      const responseTime = Date.now() - startTime;

      // Large team analytics should be under 1 second
      expect(responseTime).toBeLessThan(1000);
    });
  });

  afterAll(async () => {
    await cleanupTestData();
    await app.close();
  });
});
```

### 2. Frontend Performance Testing

```typescript
// packages/frontend/src/__tests__/performance/dashboard.performance.test.tsx
import React from 'react';
import { render, screen } from '@testing-library/react';
import { PerformanceObserver } from 'perf_hooks';
import { DashboardScreen } from '../pages/DashboardScreen/DashboardScreen';

describe('Dashboard Performance Tests', () => {
  it('should render within performance budget', async () => {
    const measure = new Promise((resolve) => {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const renderEntry = entries.find(entry => entry.name === 'render');
        if (renderEntry) {
          observer.disconnect();
          resolve(renderEntry.duration);
        }
      });
      observer.observe({ entryTypes: ['measure'] });
    });

    performance.mark('render-start');

    render(<DashboardScreen />);

    performance.mark('render-end');
    performance.measure('render', 'render-start', 'render-end');

    const renderTime = await measure;

    // Initial render should be under 100ms
    expect(renderTime).toBeLessThan(100);
  });

  it('should handle large datasets efficiently', async () => {
    const largeDataset = {
      weeklyData: Array.from({ length: 365 }, (_, i) => ({
        day: i,
        focusTime: Math.random() * 480, // 0-8 hours
        goal: 300,
      })),
    };

    const measure = new Promise((resolve) => {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const chartEntry = entries.find(entry => entry.name === 'chart-render');
        if (chartEntry) {
          observer.disconnect();
          resolve(chartEntry.duration);
        }
      });
      observer.observe({ entryTypes: ['measure'] });
    });

    performance.mark('chart-start');

    // Render chart with large dataset
    render(<WeeklyBarChart data={largeDataset.weeklyData} />);

    performance.mark('chart-end');
    performance.measure('chart-render', 'chart-start', 'chart-end');

    const chartTime = await measure;

    // Chart rendering should be under 200ms even with large datasets
    expect(chartTime).toBeLessThan(200);
  });
});
```

### 3. Memory Usage Testing

```typescript
// packages/frontend/src/__tests__/memory/dashboard.memory.test.tsx
describe('Dashboard Memory Tests', () => {
  it('should not have memory leaks on repeated navigation', async () => {
    const initialMemory = process.memoryUsage().heapUsed;

    // Simulate multiple dashboard visits
    for (let i = 0; i < 100; i++) {
      const { unmount } = render(<DashboardScreen />);
      unmount();

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
    }

    const finalMemory = process.memoryUsage().heapUsed;
    const memoryIncrease = finalMemory - initialMemory;

    // Memory increase should be minimal (< 10MB)
    expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024);
  });

  it('should clean up event listeners', () => {
    const { unmount } = render(<DashboardScreen />);

    // Check if window event listeners were added
    const initialListeners = window.addEventListener.mock.calls.length;

    unmount();

    // Event listeners should be cleaned up
    // This is a simplified test - real implementation would track specific listeners
    expect(window.removeEventListener).toHaveBeenCalledTimes(initialListeners);
  });
});
```

---

## 🔒 Security Testing

### 1. Authentication & Authorization Tests

```typescript
// packages/backend/src/analytics/__tests__/security.test.ts
describe('Analytics Security Tests', () => {
  it('should reject requests without authentication', async () => {
    await request(app.getHttpServer())
      .get('/api/analytics/focus')
      .expect(401);

    await request(app.getHttpServer())
      .get('/api/analytics/wellness')
      .expect(401);

    await request(app.getHttpServer())
      .get('/api/analytics/teams/team-123')
      .expect(401);
  });

  it('should reject requests with invalid tokens', async () => {
    await request(app.getHttpServer())
      .get('/api/analytics/focus')
      .set('Authorization', 'Bearer invalid-token')
      .expect(401);
  });

  it('should enforce team membership for team analytics', async () => {
    const userToken = await getAuthToken('user@example.com');
    const otherTeamId = 'other-team-456';

    await request(app.getHttpServer())
      .get(`/api/analytics/teams/${otherTeamId}`)
      .set('Authorization', `Bearer ${userToken}`)
      .expect(403);
  });

  it('should prevent SQL injection in date parameters', async () => {
    const maliciousDate = "2024-01-01'; DROP TABLE users; --";
    const userToken = await getAuthToken('user@example.com');

    const response = await request(app.getHttpServer())
      .get('/api/analytics/focus')
      .query({ startDate: maliciousDate })
      .set('Authorization', `Bearer ${userToken}`);

    // Should return 400 for invalid date format, not execute SQL
    expect(response.status).toBe(400);
    expect(response.body.message).toContain('Invalid date format');
  });

  it('should sanitize team IDs', async () => {
    const maliciousTeamId = "team-123'; DROP TABLE teams; --";
    const userToken = await getAuthToken('user@example.com');

    const response = await request(app.getHttpServer())
      .get(`/api/analytics/teams/${maliciousTeamId}`)
      .set('Authorization', `Bearer ${userToken}`);

    // Should return 404 for invalid team ID, not execute SQL
    expect(response.status).toBe(404);
  });
});
```

### 2. Data Validation Tests

```typescript
// packages/backend/src/analytics/__tests__/validation.test.ts
describe('Analytics Data Validation', () => {
  it('should validate date format', async () => {
    const userToken = await getAuthToken('user@example.com');

    const invalidDates = [
      'not-a-date',
      '2024-13-01', // Invalid month
      '2024-01-32', // Invalid day
      '2024-01-01T25:00:00', // Invalid hour
    ];

    for (const invalidDate of invalidDates) {
      await request(app.getHttpServer())
        .get('/api/analytics/focus')
        .query({ startDate: invalidDate })
        .set('Authorization', `Bearer ${userToken}`)
        .expect(400);
    }
  });

  it('should limit date range to prevent excessive queries', async () => {
    const userToken = await getAuthToken('user@example.com');

    // Request more than 2 years of data
    const response = await request(app.getHttpServer())
      .get('/api/analytics/focus')
      .query({
        startDate: '2020-01-01T00:00:00.000Z',
        endDate: '2024-12-31T23:59:59.999Z'
      })
      .set('Authorization', `Bearer ${userToken}`)
      .expect(400);

    expect(response.body.message).toContain('Date range too large');
  });

  it('should rate limit analytics endpoints', async () => {
    const userToken = await getAuthToken('user@example.com');

    // Make rapid requests
    const requests = Array.from({ length: 100 }, () =>
      request(app.getHttpServer())
        .get('/api/analytics/focus')
        .set('Authorization', `Bearer ${userToken}`)
    );

    const responses = await Promise.allSettled(requests);

    // Should hit rate limit after some requests
    const rateLimitedResponses = responses.filter(
      response => response.status === 429
    );

    expect(rateLimitedResponses.length).toBeGreaterThan(0);
  });
});
```

---

## 📊 Test Configuration & Setup

### 1. Jest Configuration

```json
// package.json scripts
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:e2e": "cypress run",
    "test:e2e:open": "cypress open",
    "test:performance": "jest --testPathPattern=performance",
    "test:security": "jest --testPathPattern=security"
  }
}
```

```javascript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
  },
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.tsx',
    '!src/serviceWorker.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90,
    },
  },
  testMatch: [
    '<rootDir>/src/**/__tests__/**/*.{ts,tsx}',
    '<rootDir>/src/**/*.{test,spec}.{ts,tsx}',
  ],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest',
  },
};
```

### 2. Cypress Configuration

```javascript
// cypress.config.ts
import { defineConfig } from 'cypress';

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    supportFile: 'cypress/support/e2e.ts',
    specPattern: 'cypress/e2e/**/*.cy.{js,jsx,ts,tsx}',
    video: true,
    screenshotOnRunFailure: true,
    viewportWidth: 1280,
    viewportHeight: 720,
    responseTimeout: 10000,
    requestTimeout: 10000,
    defaultCommandTimeout: 10000,
  },
  env: {
    apiUrl: 'http://localhost:3001',
  },
});
```

### 3. Test Data Setup

```typescript
// cypress/support/test-data.ts
export const createTestUser = async () => {
  // Create test user via API
  const response = await cy.request({
    method: 'POST',
    url: `${Cypress.env('apiUrl')}/auth/register`,
    body: {
      email: 'test@optomatica.com',
      password: 'password123',
      firstName: 'Test',
      lastName: 'User',
    },
  });

  return response.body;
};

export const createTestSessions = async (userId: string, count: number = 50) => {
  // Create test focus sessions
  const sessions = [];
  const now = new Date();

  for (let i = 0; i < count; i++) {
    const sessionDate = new Date(now.getTime() - (i * 24 * 60 * 60 * 1000));
    sessions.push({
      userId,
      type: 'POMODORO',
      duration: 25,
      startTime: sessionDate,
      endTime: new Date(sessionDate.getTime() + 25 * 60 * 1000),
      completed: true,
      quality: Math.floor(Math.random() * 5) + 1,
    });
  }

  // Bulk insert sessions
  await cy.request({
    method: 'POST',
    url: `${Cypress.env('apiUrl')}/test-data/sessions`,
    body: { sessions },
  });
};
```

---

## 🚀 Continuous Integration Testing

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  backend-tests:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: packages/backend/package-lock.json

    - name: Install dependencies
      working-directory: ./packages/backend
      run: npm ci

    - name: Run database migrations
      working-directory: ./packages/backend
      run: npm run prisma:migrate:test

    - name: Run unit tests
      working-directory: ./packages/backend
      run: npm run test:cov

    - name: Run integration tests
      working-directory: ./packages/backend
      run: npm run test:integration

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./packages/backend/coverage/lcov.info

  frontend-tests:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: packages/frontend/package-lock.json

    - name: Install dependencies
      working-directory: ./packages/frontend
      run: npm ci

    - name: Run unit tests
      working-directory: ./packages/frontend
      run: npm run test:cov

    - name: Build application
      working-directory: ./packages/frontend
      run: npm run build

    - name: Run E2E tests
      working-directory: ./packages/frontend
      run: npm run test:e2e:ci

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./packages/frontend/coverage/lcov.info

  performance-tests:
    runs-on: ubuntu-latest
    needs: [backend-tests, frontend-tests]

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'

    - name: Install dependencies
      run: |
        cd packages/backend && npm ci
        cd ../frontend && npm ci

    - name: Start backend
      working-directory: ./packages/backend
      run: npm run start:test &
      env:
        NODE_ENV: test
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test

    - name: Start frontend
      working-directory: ./packages/frontend
      run: npm run start:test &

    - name: Run performance tests
      working-directory: ./packages/backend
      run: npm run test:performance

    - name: Run Lighthouse CI
      run: |
        npm install -g @lhci/cli@0.12.x
        lhci autorun
      env:
        LHCI_GITHUB_APP_TOKEN: ${{ secrets.LHCI_GITHUB_APP_TOKEN }}
```

---

## ✅ Testing Success Criteria

### Coverage Requirements
- ✅ **Backend**: 95%+ code coverage for analytics services
- ✅ **Frontend**: 90%+ component coverage
- ✅ **API Endpoints**: 100% integration test coverage
- ✅ **User Journeys**: 80%+ critical path E2E coverage

### Performance Requirements
- ✅ **API Response Time**: <200ms average
- ✅ **Dashboard Load**: <2 seconds initial load
- ✅ **Chart Rendering**: <100ms for typical datasets
- ✅ **Memory Usage**: <50MB for dashboard components

### Accessibility Requirements
- ✅ **WCAG 2.1 AA**: Full compliance
- ✅ **Keyboard Navigation**: 100% interactive elements accessible
- ✅ **Screen Reader**: Complete compatibility with NVDA/JAWS
- ✅ **Color Contrast**: All text elements meet 4.5:1 ratio

### Security Requirements
- ✅ **Authentication**: All endpoints properly protected
- ✅ **Authorization**: Team data access controls enforced
- ✅ **Input Validation**: All user inputs sanitized
- ✅ **Rate Limiting**: API endpoints protected from abuse

---

**Last Updated**: 2025-01-24
**Status**: Ready for Implementation
**Dependencies**: Complete feature implementation
**Execution Order**: Unit → Integration → E2E → Performance → Security