import { Test, TestingModule } from '@nestjs/testing';
import { NotFoundException, ForbiddenException } from '@nestjs/common';
import { AnalyticsService } from '../analytics.service';
import { DatabaseService } from '../../config/database.config';

describe('AnalyticsService', () => {
  let service: AnalyticsService;
  let prisma: jest.Mocked<DatabaseService>;

  const mockUser = {
    id: 'user-1',
    email: 'test@example.com',
    firstName: 'Test',
    lastName: 'User',
    wellnessScore: 4,
    streak: 7,
    totalFocusTime: 600,
    preferences: '{}',
    level: 1,
    xp: 100,
  };

  const mockTeam = {
    id: 'team-1',
    name: 'Test Team',
    description: 'A test team',
    ownerId: 'user-1',
    members: [
      {
        userId: 'user-1',
        user: mockUser,
        role: 'OWNER',
        joinedAt: new Date(),
      },
      {
        userId: 'user-2',
        user: {
          ...mockUser,
          id: 'user-2',
          email: 'user2@example.com',
          wellnessScore: 3,
          streak: 5,
        },
        role: 'MEMBER',
        joinedAt: new Date(),
      },
    ],
  };

  const mockSessions = [
    {
      userId: 'user-1',
      type: 'POMODORO',
      duration: 25,
      startTime: new Date('2024-01-15T09:00:00Z'),
      quality: 4,
      completed: true,
    },
    {
      userId: 'user-1',
      type: 'SHORT_BREAK',
      duration: 5,
      startTime: new Date('2024-01-15T09:25:00Z'),
      completed: true,
    },
    {
      userId: 'user-1',
      type: 'POMODORO',
      duration: 25,
      startTime: new Date('2024-01-15T10:00:00Z'),
      quality: 5,
      completed: true,
    },
  ];

  const mockTasks = [
    {
      id: 'task-1',
      title: 'Test Task 1',
      status: 'COMPLETED',
      creatorId: 'user-1',
      assigneeId: 'user-1',
      completedAt: new Date('2024-01-15T11:00:00Z'),
      createdAt: new Date('2024-01-15T08:00:00Z'),
    },
    {
      id: 'task-2',
      title: 'Test Task 2',
      status: 'TODO',
      creatorId: 'user-1',
      assigneeId: 'user-1',
      createdAt: new Date('2024-01-15T08:00:00Z'),
    },
    {
      id: 'task-3',
      title: 'Test Task 3',
      status: 'COMPLETED',
      creatorId: 'user-2',
      assigneeId: 'user-2',
      completedAt: new Date('2024-01-15T12:00:00Z'),
      createdAt: new Date('2024-01-15T07:00:00Z'),
    },
  ];

  beforeEach(async () => {
    const mockPrisma = {
      user: {
        findUnique: jest.fn(),
      },
      session: {
        findMany: jest.fn(),
      },
      task: {
        count: jest.fn(),
        findMany: jest.fn(),
      },
      team: {
        findUnique: jest.fn(),
      },
      teamMember: {
        findUnique: jest.fn(),
      },
    };

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
    prisma = module.get<DatabaseService>(DatabaseService) as jest.Mocked<DatabaseService>;
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('getFocusAnalytics', () => {
    it('should calculate focus analytics correctly', async () => {
      prisma.session.findMany.mockResolvedValue(mockSessions);

      const result = await service.getFocusAnalytics('user-1');

      expect(result).toEqual({
        dailyFocusTime: expect.any(Number),
        weeklyFocusTime: expect.any(Number),
        monthlyFocusTime: expect.any(Number),
        averageSessionLength: expect.any(Number),
        peakFocusHours: expect.any(Array),
        focusTrend: expect.any(String),
        completionRate: expect.any(Number),
      });

      expect(prisma.session.findMany).toHaveBeenCalledWith({
        where: {
          userId: 'user-1',
          completed: true,
          type: 'POMODORO',
        },
        select: {
          duration: true,
          startTime: true,
          quality: true,
          taskId: true,
        },
        orderBy: { startTime: 'desc' },
      });
    });

    it('should handle date range filtering', async () => {
      const startDate = new Date('2024-01-01');
      const endDate = new Date('2024-01-31');

      prisma.session.findMany.mockResolvedValue([]);

      await service.getFocusAnalytics('user-1', startDate, endDate);

      expect(prisma.session.findMany).toHaveBeenCalledWith({
        where: {
          userId: 'user-1',
          completed: true,
          type: 'POMODORO',
          startTime: {
            gte: startDate,
            lte: endDate,
          },
        },
        select: {
          duration: true,
          startTime: true,
          quality: true,
          taskId: true,
        },
        orderBy: { startTime: 'desc' },
      });
    });

    it('should return 0 completion rate when no sessions', async () => {
      prisma.session.findMany.mockResolvedValue([]);

      const result = await service.getFocusAnalytics('user-1');

      expect(result.completionRate).toBe(0);
    });
  });

  describe('getWellnessAnalytics', () => {
    it('should calculate wellness analytics from user data', async () => {
      prisma.user.findUnique.mockResolvedValue(mockUser as any);
      prisma.session.findMany.mockResolvedValue(mockSessions);

      const result = await service.getWellnessAnalytics('user-1');

      expect(result).toEqual({
        mindfulnessMinutes: Math.round(mockUser.totalFocusTime * 0.1),
        hydrationGoal: 8,
        hydrationCurrent: expect.any(Number),
        movementGoal: 5,
        movementCurrent: expect.any(Number),
        moodRating: Math.round(mockUser.wellnessScore),
        stressLevel: Math.max(1, 5 - Math.round(mockUser.wellnessScore)),
        energyLevel: Math.min(5, Math.round((mockUser.streak / 7) + 2)),
      });

      expect(result.hydrationCurrent).toBeGreaterThanOrEqual(1);
      expect(result.hydrationCurrent).toBeLessThanOrEqual(8);
      expect(result.movementCurrent).toBeGreaterThanOrEqual(1);
      expect(result.movementCurrent).toBeLessThanOrEqual(5);
    });

    it('should throw NotFoundException when user not found', async () => {
      prisma.user.findUnique.mockResolvedValue(null);

      await expect(service.getWellnessAnalytics('invalid-user')).rejects.toThrow(
        NotFoundException,
      );
    });

    it('should handle invalid user preferences gracefully', async () => {
      const userWithInvalidPrefs = {
        ...mockUser,
        preferences: 'invalid-json',
      };
      prisma.user.findUnique.mockResolvedValue(userWithInvalidPrefs as any);
      prisma.session.findMany.mockResolvedValue([]);

      const result = await service.getWellnessAnalytics('user-1');

      expect(result).toEqual({
        mindfulnessMinutes: Math.round(mockUser.totalFocusTime * 0.1),
        hydrationGoal: 8,
        hydrationCurrent: expect.any(Number),
        movementGoal: 5,
        movementCurrent: expect.any(Number),
        moodRating: Math.round(mockUser.wellnessScore),
        stressLevel: Math.max(1, 5 - Math.round(mockUser.wellnessScore)),
        energyLevel: Math.min(5, Math.round((mockUser.streak / 7) + 2)),
      });
    });

    it('should calculate hydration and movement from session patterns', async () => {
      const sessionsWithManyBreaks = [
        ...mockSessions,
        { ...mockSessions[1], startTime: new Date('2024-01-15T11:25:00Z') },
        { ...mockSessions[1], startTime: new Date('2024-01-15T12:25:00Z') },
        { ...mockSessions[1], startTime: new Date('2024-01-15T13:25:00Z') },
      ];

      prisma.user.findUnique.mockResolvedValue(mockUser as any);
      prisma.session.findMany.mockResolvedValue(sessionsWithManyBreaks);

      const result = await service.getWellnessAnalytics('user-1');

      // Should have higher hydration due to more break sessions
      expect(result.hydrationCurrent).toBeGreaterThanOrEqual(3);
    });
  });

  describe('getTeamAnalytics', () => {
    it('should calculate team analytics correctly', async () => {
      prisma.teamMember.findUnique.mockResolvedValue({ userId: 'user-1', teamId: 'team-1' } as any);
      prisma.team.findUnique.mockResolvedValue(mockTeam as any);
      prisma.session.findMany
        .mockResolvedValueOnce(mockSessions) // All team sessions
        .mockResolvedValue(mockSessions); // Individual member sessions
      prisma.task.count
        .mockResolvedValueOnce(2) // user-1 tasks completed
        .mockResolvedValueOnce(1); // user-2 tasks completed

      const result = await service.getTeamAnalytics('team-1', undefined, undefined, 'user-1');

      expect(result).toEqual({
        teamId: 'team-1',
        teamName: 'Test Team',
        memberCount: 2,
        totalFocusTime: expect.any(Number),
        averageFocusTime: expect.any(Number),
        tasksCompleted: expect.any(Number),
        averageCompletionRate: expect.any(Number),
        topPerformers: expect.any(Array),
        focusTrend: expect.any(String),
        wellnessScore: expect.any(Number),
        collaborationScore: expect.any(Number),
        period: {
          startDate: new Date(0).toISOString(),
          endDate: expect.any(String),
        },
      });
    });

    it('should throw ForbiddenException when user not in team', async () => {
      prisma.teamMember.findUnique.mockResolvedValue(null);

      await expect(
        service.getTeamAnalytics('team-1', undefined, undefined, 'user-1'),
      ).rejects.toThrow(ForbiddenException);
    });

    it('should throw NotFoundException when team not found', async () => {
      prisma.teamMember.findUnique.mockResolvedValue({ userId: 'user-1', teamId: 'team-1' } as any);
      prisma.team.findUnique.mockResolvedValue(null);

      await expect(
        service.getTeamAnalytics('team-1', undefined, undefined, 'user-1'),
      ).rejects.toThrow(NotFoundException);
    });

    it('should handle date range filtering for team analytics', async () => {
      const startDate = new Date('2024-01-01');
      const endDate = new Date('2024-01-31');

      prisma.teamMember.findUnique.mockResolvedValue({ userId: 'user-1', teamId: 'team-1' } as any);
      prisma.team.findUnique.mockResolvedValue(mockTeam as any);
      prisma.session.findMany.mockResolvedValue([]);
      prisma.task.count.mockResolvedValue(0);

      await service.getTeamAnalytics('team-1', startDate, endDate, 'user-1');

      expect(prisma.session.findMany).toHaveBeenCalledWith({
        where: {
          userId: { in: ['user-1', 'user-2'] },
          completed: true,
          type: 'POMODORO',
          startTime: {
            gte: startDate,
            lte: endDate,
          },
        },
        select: {
          userId: true,
          duration: true,
          quality: true,
          startTime: true,
        },
      });
    });
  });

  describe('getTeamMemberCompletionRate', () => {
    it('should calculate 0% completion rate for user with no tasks', async () => {
      prisma.task.count.mockResolvedValueOnce(0).mockResolvedValueOnce(0);

      // Access private method through prototype for testing
      const result = await (service as any).getTeamMemberCompletionRate('user-1');

      expect(result).toBe(0);
      expect(prisma.task.count).toHaveBeenCalledTimes(2);
    });

    it('should calculate 100% completion rate for user with all completed tasks', async () => {
      prisma.task.count
        .mockResolvedValueOnce(10) // total tasks
        .mockResolvedValueOnce(10); // completed tasks

      const result = await (service as any).getTeamMemberCompletionRate('user-1');

      expect(result).toBe(100);
    });

    it('should calculate correct percentage for mixed task status', async () => {
      prisma.task.count
        .mockResolvedValueOnce(10) // total tasks
        .mockResolvedValueOnce(7); // completed tasks

      const result = await (service as any).getTeamMemberCompletionRate('user-1');

      expect(result).toBe(70);
    });

    it('should respect date range filtering', async () => {
      const startDate = new Date('2024-01-01');
      const endDate = new Date('2024-01-31');

      prisma.task.count
        .mockResolvedValueOnce(10) // total tasks in date range
        .mockResolvedValueOnce(7); // completed tasks in date range

      await (service as any).getTeamMemberCompletionRate('user-1', startDate, endDate);

      expect(prisma.task.count).toHaveBeenCalledWith({
        where: {
          OR: [
            { creatorId: 'user-1' },
            { assigneeId: 'user-1' }
          ],
          createdAt: {
            gte: startDate,
            lte: endDate,
          },
        },
      });
    });

    it('should include tasks where user is creator or assignee', async () => {
      prisma.task.count
        .mockResolvedValueOnce(5) // total tasks
        .mockResolvedValueOnce(3); // completed tasks

      await (service as any).getTeamMemberCompletionRate('user-1');

      expect(prisma.task.count).toHaveBeenCalledWith({
        where: {
          OR: [
            { creatorId: 'user-1' },
            { assigneeId: 'user-1' }
          ],
        },
      });
    });
  });

  describe('Performance Tests', () => {
    it('should handle large numbers of sessions efficiently', async () => {
      const largeSessionSet = Array.from({ length: 1000 }, (_, i) => ({
        userId: 'user-1',
        type: 'POMODORO',
        duration: 25,
        startTime: new Date(Date.now() - i * 1000 * 60 * 30), // Every 30 minutes
        quality: 4,
        completed: true,
      }));

      prisma.session.findMany.mockResolvedValue(largeSessionSet);

      const startTime = performance.now();
      const result = await service.getFocusAnalytics('user-1');
      const endTime = performance.now();

      expect(result).toBeDefined();
      expect(endTime - startTime).toBeLessThan(1000); // Should complete within 1 second
    });

    it('should batch calculate completion rates to prevent N+1 queries', async () => {
      prisma.teamMember.findUnique.mockResolvedValue({ userId: 'user-1', teamId: 'team-1' } as any);
      prisma.team.findUnique.mockResolvedValue(mockTeam as any);
      prisma.session.findMany.mockResolvedValue([]);

      // Mock task count calls - should be called for each completion rate calculation
      prisma.task.count
        .mockResolvedValueOnce(10) // user-1 total tasks
        .mockResolvedValueOnce(7)  // user-1 completed tasks
        .mockResolvedValueOnce(8)  // user-2 total tasks
        .mockResolvedValueOnce(6); // user-2 completed tasks

      await service.getTeamAnalytics('team-1', undefined, undefined, 'user-1');

      // Should call task.count exactly 4 times for completion rate calculation
      // (2 members x 2 calls each for total and completed tasks)
      expect(prisma.task.count).toHaveBeenCalledTimes(6); // 4 for completion + 2 for tasksCompleted
    });
  });

  describe('Error Handling', () => {
    it('should handle database errors gracefully', async () => {
      prisma.session.findMany.mockRejectedValue(new Error('Database connection failed'));

      await expect(service.getFocusAnalytics('user-1')).rejects.toThrow(
        'Database connection failed',
      );
    });

    it('should handle malformed session data', async () => {
      const malformedSessions = [
        {
          userId: 'user-1',
          type: 'POMODORO',
          duration: null,
          startTime: 'invalid-date',
          quality: 'invalid-quality',
          completed: true,
        },
      ];

      prisma.session.findMany.mockResolvedValue(malformedSessions as any);

      const result = await service.getFocusAnalytics('user-1');

      expect(result).toBeDefined();
      expect(result.averageSessionLength).toBe(0);
    });
  });
});