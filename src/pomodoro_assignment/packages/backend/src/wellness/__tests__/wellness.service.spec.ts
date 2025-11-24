import { Test, TestingModule } from '@nestjs/testing';
import { NotFoundException, ConflictException } from '@nestjs/common';
import { WellnessService } from '../wellness.service';
import { DatabaseService } from '../../config/database.config';
import {
  CreateWellnessEntryDto,
  CreateWellnessReminderDto,
  CreateWellnessGoalDto,
  UpdateWellnessEntryDto,
  WellnessReminderType,
  WellnessGoalCategory,
  WellnessGoalPeriod
} from '../dto';

describe('WellnessService', () => {
  let service: WellnessService;
  let prisma: DatabaseService;

  const mockPrisma = {
    wellnessEntry: {
      findFirst: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
      findMany: jest.fn(),
      count: jest.fn(),
    },
    wellnessReminder: {
      findMany: jest.fn(),
      findFirst: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
    },
    wellnessGoal: {
      findMany: jest.fn(),
      findFirst: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
    },
    user: {
      findUnique: jest.fn(),
    },
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        WellnessService,
        {
          provide: DatabaseService,
          useValue: mockPrisma,
        },
      ],
    }).compile();

    service = module.get<WellnessService>(WellnessService);
    prisma = module.get<DatabaseService>(DatabaseService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('getTodayWellnessEntry', () => {
    it('should return today wellness entry if it exists', async () => {
      const userId = 'user1';
      const mockEntry = {
        id: 'entry1',
        userId,
        hydrationGlasses: 6,
        hydrationGoal: 8,
        movementBreaks: 3,
        movementMinutes: 15,
        moodRating: 4,
        stressLevel: 2,
        energyLevel: 4,
        meditationMinutes: 10,
        breathingExercises: 2,
        mindfulnessSessions: 1,
        postureChecks: 4,
        eyeRestBreaks: 2,
        date: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrisma.wellnessEntry.findFirst.mockResolvedValue(mockEntry);

      const result = await service.getTodayWellnessEntry(userId);

      expect(prisma.wellnessEntry.findFirst).toHaveBeenCalledWith({
        where: {
          userId,
          date: {
            gte: expect.any(Date),
            lt: expect.any(Date),
          },
        },
      });
      expect(result).toHaveProperty('hydrationProgress');
      expect(result).toHaveProperty('wellnessScore');
    });

    it('should return null if no entry exists for today', async () => {
      const userId = 'user1';

      mockPrisma.wellnessEntry.findFirst.mockResolvedValue(null);

      const result = await service.getTodayWellnessEntry(userId);

      expect(result).toBeNull();
    });
  });

  describe('createOrUpdateWellnessEntry', () => {
    it('should create a new wellness entry if none exists for today', async () => {
      const userId = 'user1';
      const createDto: CreateWellnessEntryDto = {
        hydrationGlasses: 6,
        hydrationGoal: 8,
        movementBreaks: 3,
        movementMinutes: 15,
        moodRating: 4,
        stressLevel: 2,
        energyLevel: 4,
        meditationMinutes: 10,
        breathingExercises: 2,
        mindfulnessSessions: 1,
        postureChecks: 4,
        eyeRestBreaks: 2,
      };

      const mockCreatedEntry = {
        id: 'entry1',
        userId,
        ...createDto,
        date: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrisma.wellnessEntry.findFirst.mockResolvedValueOnce(null);
      mockPrisma.wellnessEntry.create.mockResolvedValue(mockCreatedEntry);

      const result = await service.createOrUpdateWellnessEntry(userId, createDto);

      expect(prisma.wellnessEntry.create).toHaveBeenCalledWith({
        data: {
          userId,
          ...createDto,
          date: expect.any(Date),
        },
      });
      expect(result).toHaveProperty('hydrationProgress');
      expect(result).toHaveProperty('wellnessScore');
    });

    it('should update existing wellness entry if one exists for today', async () => {
      const userId = 'user1';
      const updateDto: CreateWellnessEntryDto = {
        hydrationGlasses: 7,
        hydrationGoal: 8,
        movementBreaks: 4,
        movementMinutes: 20,
        moodRating: 5,
        stressLevel: 1,
        energyLevel: 5,
        meditationMinutes: 15,
        breathingExercises: 3,
        mindfulnessSessions: 2,
        postureChecks: 6,
        eyeRestBreaks: 3,
      };

      const mockExistingEntry = {
        id: 'entry1',
        userId,
        date: new Date(),
      };

      const mockUpdatedEntry = {
        ...mockExistingEntry,
        ...updateDto,
        updatedAt: new Date(),
      };

      mockPrisma.wellnessEntry.findFirst.mockResolvedValueOnce(mockExistingEntry);
      mockPrisma.wellnessEntry.update.mockResolvedValue(mockUpdatedEntry);

      const result = await service.createOrUpdateWellnessEntry(userId, updateDto);

      expect(prisma.wellnessEntry.update).toHaveBeenCalledWith({
        where: { id: 'entry1' },
        data: {
          ...updateDto,
          date: expect.any(Date),
        },
      });
      expect(result).toHaveProperty('hydrationProgress');
      expect(result).toHaveProperty('wellnessScore');
    });
  });

  describe('createWellnessReminder', () => {
    it('should create a new wellness reminder', async () => {
      const userId = 'user1';
      const createDto: CreateWellnessReminderDto = {
        type: WellnessReminderType.HYDRATION,
        enabled: true,
        frequency: 120,
        startTime: '09:00',
        endTime: '18:00',
        weekdays: [1, 2, 3, 4, 5],
      };

      const mockReminder = {
        id: 'reminder1',
        userId,
        ...createDto,
        weekdays: JSON.stringify(createDto.weekdays),
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrisma.wellnessReminder.findFirst.mockResolvedValue(null);
      mockPrisma.wellnessReminder.create.mockResolvedValue(mockReminder);

      const result = await service.createWellnessReminder(userId, createDto);

      expect(prisma.wellnessReminder.create).toHaveBeenCalledWith({
        data: {
          userId,
          ...createDto,
          weekdays: JSON.stringify(createDto.weekdays),
        },
      });
      expect(result.weekdays).toEqual(createDto.weekdays);
    });

    it('should throw ConflictException if reminder type already exists', async () => {
      const userId = 'user1';
      const createDto: CreateWellnessReminderDto = {
        type: WellnessReminderType.HYDRATION,
        enabled: true,
        frequency: 120,
        startTime: '09:00',
        endTime: '18:00',
        weekdays: [1, 2, 3, 4, 5],
      };

      const existingReminder = {
        id: 'reminder1',
        userId,
        type: WellnessReminderType.HYDRATION,
      };

      mockPrisma.wellnessReminder.findFirst.mockResolvedValue(existingReminder);

      await expect(service.createWellnessReminder(userId, createDto))
        .rejects.toThrow(ConflictException);
    });
  });

  describe('createWellnessGoal', () => {
    it('should create a new wellness goal', async () => {
      const userId = 'user1';
      const createDto: CreateWellnessGoalDto = {
        category: WellnessGoalCategory.HYDRATION,
        targetValue: 8,
        period: WellnessGoalPeriod.DAILY,
        active: true,
      };

      const mockGoal = {
        id: 'goal1',
        userId,
        ...createDto,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrisma.wellnessGoal.findFirst.mockResolvedValue(null);
      mockPrisma.wellnessGoal.create.mockResolvedValue(mockGoal);

      const result = await service.createWellnessGoal(userId, createDto);

      expect(prisma.wellnessGoal.create).toHaveBeenCalledWith({
        data: { userId, ...createDto },
      });
      expect(result).toHaveProperty('currentProgress');
      expect(result).toHaveProperty('progressPercentage');
    });

    it('should throw ConflictException if goal with same category and period exists', async () => {
      const userId = 'user1';
      const createDto: CreateWellnessGoalDto = {
        category: WellnessGoalCategory.HYDRATION,
        targetValue: 8,
        period: WellnessGoalPeriod.DAILY,
        active: true,
      };

      const existingGoal = {
        id: 'goal1',
        userId,
        category: WellnessGoalCategory.HYDRATION,
        period: WellnessGoalPeriod.DAILY,
      };

      mockPrisma.wellnessGoal.findFirst.mockResolvedValue(existingGoal);

      await expect(service.createWellnessGoal(userId, createDto))
        .rejects.toThrow(ConflictException);
    });
  });

  describe('updateWellnessEntryByDate', () => {
    it('should update wellness entry for specific date', async () => {
      const userId = 'user1';
      const targetDate = new Date('2024-01-15');
      const updateDto: UpdateWellnessEntryDto = {
        hydrationGlasses: 7,
        moodRating: 5,
      };

      const mockEntry = {
        id: 'entry1',
        userId,
        date: targetDate,
      };

      const mockUpdatedEntry = {
        ...mockEntry,
        ...updateDto,
        updatedAt: new Date(),
      };

      mockPrisma.wellnessEntry.findFirst.mockResolvedValue(mockEntry);
      mockPrisma.wellnessEntry.update.mockResolvedValue(mockUpdatedEntry);

      const result = await service.updateWellnessEntryByDate(userId, targetDate, updateDto);

      expect(prisma.wellnessEntry.findFirst).toHaveBeenCalledWith({
        where: {
          userId,
          date: {
            gte: expect.any(Date),
            lt: expect.any(Date),
          },
        },
      });
      expect(prisma.wellnessEntry.update).toHaveBeenCalledWith({
        where: { id: 'entry1' },
        data: updateDto,
      });
      expect(result).toHaveProperty('hydrationProgress');
      expect(result).toHaveProperty('wellnessScore');
    });

    it('should throw NotFoundException if no entry exists for the date', async () => {
      const userId = 'user1';
      const targetDate = new Date('2024-01-15');
      const updateDto: UpdateWellnessEntryDto = {
        hydrationGlasses: 7,
      };

      mockPrisma.wellnessEntry.findFirst.mockResolvedValue(null);

      await expect(service.updateWellnessEntryByDate(userId, targetDate, updateDto))
        .rejects.toThrow(NotFoundException);
    });
  });

  describe('deleteWellnessEntryByDate', () => {
    it('should delete wellness entry for specific date', async () => {
      const userId = 'user1';
      const targetDate = new Date('2024-01-15');
      const mockEntry = {
        id: 'entry1',
        userId,
        date: targetDate,
      };

      mockPrisma.wellnessEntry.findFirst.mockResolvedValue(mockEntry);
      mockPrisma.wellnessEntry.delete.mockResolvedValue(mockEntry);

      const result = await service.deleteWellnessEntryByDate(userId, targetDate);

      expect(prisma.wellnessEntry.findFirst).toHaveBeenCalled();
      expect(prisma.wellnessEntry.delete).toHaveBeenCalledWith({
        where: { id: 'entry1' },
      });
      expect(result).toEqual({ message: 'Wellness entry deleted successfully' });
    });

    it('should throw NotFoundException if no entry exists for the date', async () => {
      const userId = 'user1';
      const targetDate = new Date('2024-01-15');

      mockPrisma.wellnessEntry.findFirst.mockResolvedValue(null);

      await expect(service.deleteWellnessEntryByDate(userId, targetDate))
        .rejects.toThrow(NotFoundException);
    });
  });

  describe('getWellnessHistory', () => {
    it('should return paginated wellness history', async () => {
      const userId = 'user1';
      const query = {
        days: 30,
        page: 1,
        limit: 10,
        sortBy: 'date',
        sortOrder: 'desc' as const,
      };

      const mockEntries = [
        {
          id: 'entry1',
          userId,
          hydrationGlasses: 6,
          date: new Date(),
          createdAt: new Date(),
          updatedAt: new Date(),
        },
        {
          id: 'entry2',
          userId,
          hydrationGlasses: 7,
          date: new Date(),
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ];

      mockPrisma.wellnessEntry.findMany.mockResolvedValue(mockEntries);
      mockPrisma.wellnessEntry.count.mockResolvedValue(2);

      const result = await service.getWellnessHistory(userId, query);

      expect(prisma.wellnessEntry.findMany).toHaveBeenCalledWith({
        where: {
          userId,
          date: expect.any(Object),
        },
        orderBy: {
          date: 'desc',
        },
        skip: 0,
        take: 10,
      });
      expect(result).toHaveProperty('entries');
      expect(result).toHaveProperty('pagination');
      expect(result.pagination).toHaveProperty('page', 1);
      expect(result.pagination).toHaveProperty('limit', 10);
      expect(result.pagination).toHaveProperty('total', 2);
    });
  });

  describe('getWellnessReminders', () => {
    it('should return all wellness reminders for a user', async () => {
      const userId = 'user1';
      const mockReminders = [
        {
          id: 'reminder1',
          userId,
          type: WellnessReminderType.HYDRATION,
          enabled: true,
          frequency: 120,
          startTime: '09:00',
          endTime: '18:00',
          weekdays: JSON.stringify([1, 2, 3, 4, 5]),
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ];

      mockPrisma.wellnessReminder.findMany.mockResolvedValue(mockReminders);

      const result = await service.getWellnessReminders(userId);

      expect(prisma.wellnessReminder.findMany).toHaveBeenCalledWith({
        where: { userId },
        orderBy: { type: 'asc' },
      });
      expect(result[0].weekdays).toEqual([1, 2, 3, 4, 5]);
    });
  });

  describe('getWellnessGoals', () => {
    it('should return all wellness goals with progress', async () => {
      const userId = 'user1';
      const mockGoals = [
        {
          id: 'goal1',
          userId,
          category: WellnessGoalCategory.HYDRATION,
          targetValue: 8,
          period: WellnessGoalPeriod.DAILY,
          active: true,
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ];

      mockPrisma.wellnessGoal.findMany.mockResolvedValue(mockGoals);
      mockPrisma.wellnessEntry.findMany.mockResolvedValue([]);

      const result = await service.getWellnessGoals(userId);

      expect(prisma.wellnessGoal.findMany).toHaveBeenCalledWith({
        where: { userId },
        orderBy: { category: 'asc' },
      });
      expect(result[0]).toHaveProperty('currentProgress');
      expect(result[0]).toHaveProperty('progressPercentage');
    });
  });

  describe('getWellnessAnalytics', () => {
    it('should return comprehensive wellness analytics', async () => {
      const userId = 'user1';
      const query = {
        days: 30,
        includeRecommendations: true,
        includeTrends: true,
      };

      const mockEntries = [
        {
          id: 'entry1',
          userId,
          hydrationGlasses: 6,
          hydrationGoal: 8,
          movementBreaks: 3,
          moodRating: 4,
          stressLevel: 2,
          energyLevel: 4,
          meditationMinutes: 10,
          sleepHours: 7,
          sleepQuality: 4,
          date: new Date(),
        },
      ];

      mockPrisma.wellnessEntry.findMany.mockResolvedValue(mockEntries);

      const result = await service.getWellnessAnalytics(userId, query);

      expect(prisma.wellnessEntry.findMany).toHaveBeenCalledWith({
        where: {
          userId,
          date: {
            gte: expect.any(Date),
            lte: expect.any(Date),
          },
        },
        orderBy: { date: 'asc' },
      });
      expect(result).toHaveProperty('userId', userId);
      expect(result).toHaveProperty('period', 30);
      expect(result).toHaveProperty('hydration');
      expect(result).toHaveProperty('movement');
      expect(result).toHaveProperty('mentalWellness');
      expect(result).toHaveProperty('sleep');
      expect(result).toHaveProperty('overall');
      expect(result).toHaveProperty('recommendations');
      expect(result).toHaveProperty('trends');
    });

    it('should return analytics without recommendations and trends when disabled', async () => {
      const userId = 'user1';
      const query = {
        days: 30,
        includeRecommendations: false,
        includeTrends: false,
      };

      mockPrisma.wellnessEntry.findMany.mockResolvedValue([]);

      const result = await service.getWellnessAnalytics(userId, query);

      expect(result).not.toHaveProperty('recommendations');
      expect(result).not.toHaveProperty('trends');
    });
  });

  describe('getWellnessSummary', () => {
    it('should return wellness summary for dashboard', async () => {
      const userId = 'user1';

      const mockTodayEntry = {
        id: 'entry1',
        userId,
        hydrationGlasses: 6,
        hydrationGoal: 8,
        movementBreaks: 3,
        moodRating: 4,
        stressLevel: 2,
        energyLevel: 4,
        date: new Date(),
      };

      const mockWeeklyEntries = [
        mockTodayEntry,
        {
          id: 'entry2',
          userId,
          hydrationGlasses: 7,
          hydrationGoal: 8,
          movementBreaks: 4,
          moodRating: 5,
          stressLevel: 1,
          energyLevel: 5,
          date: new Date(),
        },
      ];

      const mockGoals = [
        {
          id: 'goal1',
          userId,
          category: WellnessGoalCategory.HYDRATION,
          targetValue: 8,
          period: WellnessGoalPeriod.DAILY,
          active: true,
        },
      ];

      jest.spyOn(service, 'getTodayWellnessEntry').mockResolvedValue(mockTodayEntry);
      jest.spyOn(service as any, 'getRecentWellnessEntries').mockResolvedValue(mockWeeklyEntries);
      jest.spyOn(service as any, 'getWellnessStreak').mockResolvedValue(5);
      jest.spyOn(service, 'getWellnessGoals').mockResolvedValue(mockGoals);

      const result = await service.getWellnessSummary(userId);

      expect(result).toHaveProperty('today');
      expect(result).toHaveProperty('weeklyAverage');
      expect(result).toHaveProperty('streak', 5);
      expect(result).toHaveProperty('goals');
      expect(result.weeklyAverage).toHaveProperty('hydrationGlasses');
    });
  });
});