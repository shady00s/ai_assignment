import { Test, TestingModule } from '@nestjs/testing';
import { AnalyticsController } from '../analytics.controller';
import { AnalyticsService } from '../analytics.service';
import { JwtAuthGuard } from '../../auth/guards/jwt-auth.guard';
import { FocusAnalyticsDto, WellnessAnalyticsDto, TeamAnalyticsDto } from '../dto';

describe('AnalyticsController', () => {
  let controller: AnalyticsController;
  let analyticsService: jest.Mocked<AnalyticsService>;

  const mockUser = {
    id: 'user-1',
    email: 'test@example.com',
  };

  const mockFocusAnalytics: FocusAnalyticsDto = {
    dailyFocusTime: 225,
    weeklyFocusTime: 1575,
    monthlyFocusTime: 6750,
    averageSessionLength: 25.5,
    peakFocusHours: [9, 10, 14],
    focusTrend: 'IMPROVING',
    completionRate: 85.5,
  };

  const mockWellnessAnalytics: WellnessAnalyticsDto = {
    mindfulnessMinutes: 60,
    hydrationGoal: 8,
    hydrationCurrent: 6,
    movementGoal: 5,
    movementCurrent: 3,
    moodRating: 4,
    stressLevel: 2,
    energyLevel: 4,
  };

  const mockTeamAnalytics: TeamAnalyticsDto = {
    teamId: 'team-1',
    teamName: 'Test Team',
    memberCount: 5,
    totalFocusTime: 3600,
    averageFocusTime: 720,
    tasksCompleted: 48,
    averageCompletionRate: 80,
    topPerformers: [],
    focusTrend: 'IMPROVING',
    wellnessScore: 82,
    collaborationScore: 75,
    period: {
      startDate: '2024-01-01T00:00:00.000Z',
      endDate: '2024-01-31T23:59:59.999Z',
    },
  };

  beforeEach(async () => {
    const mockAnalyticsService = {
      getFocusAnalytics: jest.fn(),
      getWellnessAnalytics: jest.fn(),
      getTeamAnalytics: jest.fn(),
    };

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
      .useValue({ canActivate: jest.fn(() => true) })
      .compile();

    controller = module.get<AnalyticsController>(AnalyticsController);
    analyticsService = module.get<AnalyticsService>(AnalyticsService) as jest.Mocked<AnalyticsService>;
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('getFocusAnalytics', () => {
    it('should return focus analytics for the current user', async () => {
      const mockRequest = { user: mockUser } as any;
      const query = { startDate: '2024-01-01T00:00:00.000Z', endDate: '2024-01-31T23:59:59.999Z' };

      analyticsService.getFocusAnalytics.mockResolvedValue(mockFocusAnalytics);

      const result = await controller.getFocusAnalytics(mockRequest, query);

      expect(result).toEqual(mockFocusAnalytics);
      expect(analyticsService.getFocusAnalytics).toHaveBeenCalledWith(
        mockUser.id,
        new Date('2024-01-01T00:00:00.000Z'),
        new Date('2024-01-31T23:59:59.999Z'),
      );
    });

    it('should handle query without date range', async () => {
      const mockRequest = { user: mockUser } as any;
      const query = {};

      analyticsService.getFocusAnalytics.mockResolvedValue(mockFocusAnalytics);

      const result = await controller.getFocusAnalytics(mockRequest, query);

      expect(result).toEqual(mockFocusAnalytics);
      expect(analyticsService.getFocusAnalytics).toHaveBeenCalledWith(
        mockUser.id,
        undefined,
        undefined,
      );
    });

    it('should handle partial date range', async () => {
      const mockRequest = { user: mockUser } as any;
      const query = { startDate: '2024-01-01T00:00:00.000Z' };

      analyticsService.getFocusAnalytics.mockResolvedValue(mockFocusAnalytics);

      await controller.getFocusAnalytics(mockRequest, query);

      expect(analyticsService.getFocusAnalytics).toHaveBeenCalledWith(
        mockUser.id,
        new Date('2024-01-01T00:00:00.000Z'),
        undefined,
      );
    });
  });

  describe('getWellnessAnalytics', () => {
    it('should return wellness analytics for the current user', async () => {
      const mockRequest = { user: mockUser } as any;
      const query = { startDate: '2024-01-01T00:00:00.000Z', endDate: '2024-01-31T23:59:59.999Z' };

      analyticsService.getWellnessAnalytics.mockResolvedValue(mockWellnessAnalytics);

      const result = await controller.getWellnessAnalytics(mockRequest, query);

      expect(result).toEqual(mockWellnessAnalytics);
      expect(analyticsService.getWellnessAnalytics).toHaveBeenCalledWith(
        mockUser.id,
        new Date('2024-01-01T00:00:00.000Z'),
        new Date('2024-01-31T23:59:59.999Z'),
      );
    });

    it('should handle query without date range', async () => {
      const mockRequest = { user: mockUser } as any;
      const query = {};

      analyticsService.getWellnessAnalytics.mockResolvedValue(mockWellnessAnalytics);

      const result = await controller.getWellnessAnalytics(mockRequest, query);

      expect(result).toEqual(mockWellnessAnalytics);
      expect(analyticsService.getWellnessAnalytics).toHaveBeenCalledWith(
        mockUser.id,
        undefined,
        undefined,
      );
    });
  });

  describe('getTeamAnalytics', () => {
    it('should return team analytics', async () => {
      const mockRequest = { user: mockUser } as any;
      const teamId = 'team-1';
      const query = { startDate: '2024-01-01T00:00:00.000Z', endDate: '2024-01-31T23:59:59.999Z' };

      analyticsService.getTeamAnalytics.mockResolvedValue(mockTeamAnalytics);

      const result = await controller.getTeamAnalytics(teamId, mockRequest, query);

      expect(result).toEqual(mockTeamAnalytics);
      expect(analyticsService.getTeamAnalytics).toHaveBeenCalledWith(
        teamId,
        new Date('2024-01-01T00:00:00.000Z'),
        new Date('2024-01-31T23:59:59.999Z'),
        mockUser.id,
      );
    });

    it('should handle query without date range', async () => {
      const mockRequest = { user: mockUser } as any;
      const teamId = 'team-1';
      const query = {};

      analyticsService.getTeamAnalytics.mockResolvedValue(mockTeamAnalytics);

      const result = await controller.getTeamAnalytics(teamId, mockRequest, query);

      expect(result).toEqual(mockTeamAnalytics);
      expect(analyticsService.getTeamAnalytics).toHaveBeenCalledWith(
        teamId,
        undefined,
        undefined,
        mockUser.id,
      );
    });

    it('should handle invalid date strings gracefully', async () => {
      const mockRequest = { user: mockUser } as any;
      const teamId = 'team-1';
      const query = { startDate: 'invalid-date', endDate: 'another-invalid-date' };

      analyticsService.getTeamAnalytics.mockResolvedValue(mockTeamAnalytics);

      const result = await controller.getTeamAnalytics(teamId, mockRequest, query);

      expect(result).toEqual(mockTeamAnalytics);
      // The dates will be passed as Invalid Date objects to the service
      expect(analyticsService.getTeamAnalytics).toHaveBeenCalledWith(
        teamId,
        expect.any(Date), // Invalid Date
        expect.any(Date), // Invalid Date
        mockUser.id,
      );
    });
  });

  describe('Input Validation', () => {
    it('should validate ISO 8601 date format', async () => {
      const mockRequest = { user: mockUser } as any;
      const validQuery = {
        startDate: '2024-01-01T00:00:00.000Z',
        endDate: '2024-01-31T23:59:59.999Z',
      };

      analyticsService.getFocusAnalytics.mockResolvedValue(mockFocusAnalytics);

      const result = await controller.getFocusAnalytics(mockRequest, validQuery);

      expect(result).toBeDefined();
      expect(analyticsService.getFocusAnalytics).toHaveBeenCalledWith(
        mockUser.id,
        new Date('2024-01-01T00:00:00.000Z'),
        new Date('2024-01-31T23:59:59.999Z'),
      );
    });

    it('should handle malformed but parsable dates', async () => {
      const mockRequest = { user: mockUser } as any;
      const query = {
        startDate: '2024-01-01', // Missing time part
        endDate: '2024-01-31',
      };

      analyticsService.getFocusAnalytics.mockResolvedValue(mockFocusAnalytics);

      const result = await controller.getFocusAnalytics(mockRequest, query);

      expect(result).toBeDefined();
      // Dates should be parsed (will be treated as midnight UTC)
      expect(analyticsService.getFocusAnalytics).toHaveBeenCalledWith(
        mockUser.id,
        new Date('2024-01-01'),
        new Date('2024-01-31'),
      );
    });
  });

  describe('Error Propagation', () => {
    it('should propagate service errors', async () => {
      const mockRequest = { user: mockUser } as any;
      const query = {};

      const error = new Error('Service unavailable');
      analyticsService.getFocusAnalytics.mockRejectedValue(error);

      await expect(controller.getFocusAnalytics(mockRequest, query)).rejects.toThrow(
        'Service unavailable',
      );
    });

    it('should handle null service responses', async () => {
      const mockRequest = { user: mockUser } as any;
      const query = {};

      analyticsService.getFocusAnalytics.mockResolvedValue(null as any);

      const result = await controller.getFocusAnalytics(mockRequest, query);

      expect(result).toBeNull();
    });
  });
});