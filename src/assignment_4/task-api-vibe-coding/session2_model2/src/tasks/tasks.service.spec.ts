import { Test, TestingModule } from '@nestjs/testing';
import { TasksService } from './tasks.service';
import { PrismaService } from '../prisma.service';
import { TaskStatus, TaskPriority } from '@prisma/client';
import { CreateTaskDto } from './dto/create-task.dto';
import { UpdateTaskDto } from './dto/update-task.dto';
import { TaskFilterDto } from './dto/task-filter.dto';
import { NotFoundException } from '@nestjs/common';

describe('TasksService', () => {
  let service: TasksService;
  let prismaService: PrismaService;

  const mockPrismaService = {
    task: {
      create: jest.fn(),
      findMany: jest.fn(),
      findUnique: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
    },
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        TasksService,
        {
          provide: PrismaService,
          useValue: mockPrismaService,
        },
      ],
    }).compile();

    service = module.get<TasksService>(TasksService);
    prismaService = module.get<PrismaService>(PrismaService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('create', () => {
    it('should create a task with default priority', async () => {
      const createTaskDto: CreateTaskDto = {
        title: 'Test Task',
        description: 'Test Description',
      };

      const expectedTask = {
        id: '1',
        title: 'Test Task',
        description: 'Test Description',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrismaService.task.create.mockResolvedValue(expectedTask);

      const result = await service.create(createTaskDto);

      expect(mockPrismaService.task.create).toHaveBeenCalledWith({
        data: {
          title: 'Test Task',
          description: 'Test Description',
          priority: TaskPriority.MEDIUM,
        },
      });
      expect(result).toEqual({
        id: '1',
        title: 'Test Task',
        description: 'Test Description',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: expectedTask.createdAt,
        updatedAt: expectedTask.updatedAt,
      });
    });

    it('should create a task with custom priority', async () => {
      const createTaskDto: CreateTaskDto = {
        title: 'High Priority Task',
        description: 'Important task',
        priority: TaskPriority.HIGH,
      };

      const expectedTask = {
        id: '2',
        title: 'High Priority Task',
        description: 'Important task',
        status: TaskStatus.PENDING,
        priority: TaskPriority.HIGH,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrismaService.task.create.mockResolvedValue(expectedTask);

      const result = await service.create(createTaskDto);

      expect(mockPrismaService.task.create).toHaveBeenCalledWith({
        data: {
          title: 'High Priority Task',
          description: 'Important task',
          priority: TaskPriority.HIGH,
        },
      });
      expect(result.priority).toBe(TaskPriority.HIGH);
    });

    it('should create a task without description', async () => {
      const createTaskDto: CreateTaskDto = {
        title: 'Simple Task',
      };

      const expectedTask = {
        id: '3',
        title: 'Simple Task',
        description: null,
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrismaService.task.create.mockResolvedValue(expectedTask);

      const result = await service.create(createTaskDto);

      expect(result.description).toBeNull();
    });
  });

  describe('findAll', () => {
    it('should return all tasks ordered by creation date descending', async () => {
      const filterDto: TaskFilterDto = {};
      const expectedTasks = [
        {
          id: '1',
          title: 'Recent Task',
          description: 'Most recent',
          status: TaskStatus.PENDING,
          priority: TaskPriority.HIGH,
          createdAt: new Date('2024-01-02'),
          updatedAt: new Date('2024-01-02'),
        },
        {
          id: '2',
          title: 'Older Task',
          description: 'Created earlier',
          status: TaskStatus.COMPLETED,
          priority: TaskPriority.LOW,
          createdAt: new Date('2024-01-01'),
          updatedAt: new Date('2024-01-01'),
        },
      ];

      mockPrismaService.task.findMany.mockResolvedValue(expectedTasks);

      const result = await service.findAll(filterDto);

      expect(mockPrismaService.task.findMany).toHaveBeenCalledWith({
        where: {},
        orderBy: {
          createdAt: 'desc',
        },
      });
      expect(result).toHaveLength(2);
      expect(result[0].title).toBe('Recent Task');
    });

    it('should filter tasks by status', async () => {
      const filterDto: TaskFilterDto = {
        status: TaskStatus.PENDING,
      };

      const expectedTasks = [
        {
          id: '1',
          title: 'Pending Task',
          description: 'Pending description',
          status: TaskStatus.PENDING,
          priority: TaskPriority.MEDIUM,
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ];

      mockPrismaService.task.findMany.mockResolvedValue(expectedTasks);

      const result = await service.findAll(filterDto);

      expect(mockPrismaService.task.findMany).toHaveBeenCalledWith({
        where: {
          status: TaskStatus.PENDING,
        },
        orderBy: {
          createdAt: 'desc',
        },
      });
      expect(result).toHaveLength(1);
      expect(result[0].status).toBe(TaskStatus.PENDING);
    });

    it('should filter tasks by priority', async () => {
      const filterDto: TaskFilterDto = {
        priority: TaskPriority.HIGH,
      };

      const expectedTasks = [
        {
          id: '1',
          title: 'High Priority Task',
          description: 'Important',
          status: TaskStatus.IN_PROGRESS,
          priority: TaskPriority.HIGH,
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ];

      mockPrismaService.task.findMany.mockResolvedValue(expectedTasks);

      const result = await service.findAll(filterDto);

      expect(mockPrismaService.task.findMany).toHaveBeenCalledWith({
        where: {
          priority: TaskPriority.HIGH,
        },
        orderBy: {
          createdAt: 'desc',
        },
      });
      expect(result).toHaveLength(1);
      expect(result[0].priority).toBe(TaskPriority.HIGH);
    });

    it('should filter tasks by both status and priority', async () => {
      const filterDto: TaskFilterDto = {
        status: TaskStatus.COMPLETED,
        priority: TaskPriority.LOW,
      };

      const expectedTasks = [
        {
          id: '1',
          title: 'Completed Low Priority Task',
          description: 'Done',
          status: TaskStatus.COMPLETED,
          priority: TaskPriority.LOW,
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ];

      mockPrismaService.task.findMany.mockResolvedValue(expectedTasks);

      const result = await service.findAll(filterDto);

      expect(mockPrismaService.task.findMany).toHaveBeenCalledWith({
        where: {
          status: TaskStatus.COMPLETED,
          priority: TaskPriority.LOW,
        },
        orderBy: {
          createdAt: 'desc',
        },
      });
      expect(result).toHaveLength(1);
    });

    it('should return empty array when no tasks match filter', async () => {
      const filterDto: TaskFilterDto = {
        status: TaskStatus.COMPLETED,
      };

      mockPrismaService.task.findMany.mockResolvedValue([]);

      const result = await service.findAll(filterDto);

      expect(result).toHaveLength(0);
      expect(result).toEqual([]);
    });
  });

  describe('findOne', () => {
    it('should return a task when found', async () => {
      const expectedTask = {
        id: '1',
        title: 'Found Task',
        description: 'Task description',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrismaService.task.findUnique.mockResolvedValue(expectedTask);

      const result = await service.findOne('1');

      expect(mockPrismaService.task.findUnique).toHaveBeenCalledWith({
        where: { id: '1' },
      });
      expect(result).toEqual(expectedTask);
    });

    it('should throw NotFoundException when task not found', async () => {
      mockPrismaService.task.findUnique.mockResolvedValue(null);

      await expect(service.findOne('999')).rejects.toThrow(
        new NotFoundException('Task with ID 999 not found')
      );

      expect(mockPrismaService.task.findUnique).toHaveBeenCalledWith({
        where: { id: '999' },
      });
    });
  });

  describe('update', () => {
    it('should update a task successfully', async () => {
      const updateTaskDto: UpdateTaskDto = {
        title: 'Updated Task',
        status: TaskStatus.IN_PROGRESS,
      };

      const existingTask = {
        id: '1',
        title: 'Original Task',
        description: 'Original description',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      const updatedTask = {
        id: '1',
        title: 'Updated Task',
        description: 'Original description',
        status: TaskStatus.IN_PROGRESS,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrismaService.task.findUnique
        .mockResolvedValueOnce(existingTask)
        .mockResolvedValueOnce(updatedTask);
      mockPrismaService.task.update.mockResolvedValue(updatedTask);

      const result = await service.update('1', updateTaskDto);

      expect(mockPrismaService.task.findUnique).toHaveBeenCalledWith({
        where: { id: '1' },
      });
      expect(mockPrismaService.task.update).toHaveBeenCalledWith({
        where: { id: '1' },
        data: updateTaskDto,
      });
      expect(result.title).toBe('Updated Task');
      expect(result.status).toBe(TaskStatus.IN_PROGRESS);
    });

    it('should update only provided fields', async () => {
      const updateTaskDto: UpdateTaskDto = {
        description: 'New description only',
      };

      const existingTask = {
        id: '1',
        title: 'Task Title',
        description: 'Original description',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      const updatedTask = {
        id: '1',
        title: 'Task Title',
        description: 'New description only',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrismaService.task.findUnique
        .mockResolvedValueOnce(existingTask)
        .mockResolvedValueOnce(updatedTask);
      mockPrismaService.task.update.mockResolvedValue(updatedTask);

      const result = await service.update('1', updateTaskDto);

      expect(mockPrismaService.task.update).toHaveBeenCalledWith({
        where: { id: '1' },
        data: { description: 'New description only' },
      });
      expect(result.title).toBe('Task Title'); // unchanged
      expect(result.description).toBe('New description only'); // changed
    });

    it('should throw NotFoundException when updating non-existent task', async () => {
      const updateTaskDto: UpdateTaskDto = {
        title: 'Updated Task',
      };

      mockPrismaService.task.findUnique.mockResolvedValue(null);

      await expect(service.update('999', updateTaskDto)).rejects.toThrow(
        new NotFoundException('Task with ID 999 not found')
      );

      expect(mockPrismaService.task.findUnique).toHaveBeenCalledWith({
        where: { id: '999' },
      });
      expect(mockPrismaService.task.update).not.toHaveBeenCalled();
    });

    it('should handle empty update DTO', async () => {
      const updateTaskDto: UpdateTaskDto = {};

      const existingTask = {
        id: '1',
        title: 'Task Title',
        description: 'Description',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrismaService.task.findUnique
        .mockResolvedValueOnce(existingTask)
        .mockResolvedValueOnce(existingTask);
      mockPrismaService.task.update.mockResolvedValue(existingTask);

      const result = await service.update('1', updateTaskDto);

      expect(mockPrismaService.task.update).toHaveBeenCalledWith({
        where: { id: '1' },
        data: {},
      });
      expect(result).toEqual(existingTask);
    });
  });

  describe('remove', () => {
    it('should delete a task successfully', async () => {
      const existingTask = {
        id: '1',
        title: 'Task to Delete',
        description: 'Will be deleted',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrismaService.task.findUnique.mockResolvedValue(existingTask);
      mockPrismaService.task.delete.mockResolvedValue(existingTask);

      await service.remove('1');

      expect(mockPrismaService.task.findUnique).toHaveBeenCalledWith({
        where: { id: '1' },
      });
      expect(mockPrismaService.task.delete).toHaveBeenCalledWith({
        where: { id: '1' },
      });
    });

    it('should throw NotFoundException when deleting non-existent task', async () => {
      mockPrismaService.task.findUnique.mockResolvedValue(null);

      await expect(service.remove('999')).rejects.toThrow(
        new NotFoundException('Task with ID 999 not found')
      );

      expect(mockPrismaService.task.findUnique).toHaveBeenCalledWith({
        where: { id: '999' },
      });
      expect(mockPrismaService.task.delete).not.toHaveBeenCalled();
    });

    it('should resolve void when deletion is successful', async () => {
      const existingTask = {
        id: '1',
        title: 'Task to Delete',
        description: 'Will be deleted',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockPrismaService.task.findUnique.mockResolvedValue(existingTask);
      mockPrismaService.task.delete.mockResolvedValue(existingTask);

      const result = service.remove('1');

      await expect(result).resolves.toBeUndefined();
    });
  });
});