import { Test, TestingModule } from '@nestjs/testing';
import { TasksController } from './tasks.controller';
import { TasksService } from './tasks.service';
import { CreateTaskDto } from './dto/create-task.dto';
import { UpdateTaskDto } from './dto/update-task.dto';
import { TaskFilterDto } from './dto/task-filter.dto';
import { TaskResponseDto } from './dto/task-response.dto';
import { TaskStatus, TaskPriority } from '@prisma/client';
import { HttpException } from '@nestjs/common';

describe('TasksController', () => {
  let controller: TasksController;
  let service: TasksService;

  const mockTasksService = {
    create: jest.fn(),
    findAll: jest.fn(),
    findOne: jest.fn(),
    update: jest.fn(),
    remove: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [TasksController],
      providers: [
        {
          provide: TasksService,
          useValue: mockTasksService,
        },
      ],
    }).compile();

    controller = module.get<TasksController>(TasksController);
    service = module.get<TasksService>(TasksService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('create', () => {
    it('should create a new task', async () => {
      const createTaskDto: CreateTaskDto = {
        title: 'New Task',
        description: 'Task description',
        priority: TaskPriority.HIGH,
      };

      const expectedResponse: TaskResponseDto = {
        id: '1',
        title: 'New Task',
        description: 'Task description',
        status: TaskStatus.PENDING,
        priority: TaskPriority.HIGH,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockTasksService.create.mockResolvedValue(expectedResponse);

      const result = await controller.create(createTaskDto);

      expect(service.create).toHaveBeenCalledWith(createTaskDto);
      expect(result).toEqual(expectedResponse);
    });

    it('should create a task with default values', async () => {
      const createTaskDto: CreateTaskDto = {
        title: 'Simple Task',
      };

      const expectedResponse: TaskResponseDto = {
        id: '2',
        title: 'Simple Task',
        description: null,
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockTasksService.create.mockResolvedValue(expectedResponse);

      const result = await controller.create(createTaskDto);

      expect(service.create).toHaveBeenCalledWith(createTaskDto);
      expect(result.title).toBe('Simple Task');
      expect(result.description).toBeNull();
    });
  });

  describe('findAll', () => {
    it('should return all tasks without filters', async () => {
      const filterDto: TaskFilterDto = {};
      const expectedResponse: TaskResponseDto[] = [
        {
          id: '1',
          title: 'Task 1',
          description: 'Description 1',
          status: TaskStatus.PENDING,
          priority: TaskPriority.HIGH,
          createdAt: new Date(),
          updatedAt: new Date(),
        },
        {
          id: '2',
          title: 'Task 2',
          description: 'Description 2',
          status: TaskStatus.COMPLETED,
          priority: TaskPriority.LOW,
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ];

      mockTasksService.findAll.mockResolvedValue(expectedResponse);

      const result = await controller.findAll(filterDto);

      expect(service.findAll).toHaveBeenCalledWith(filterDto);
      expect(result).toHaveLength(2);
      expect(result[0].title).toBe('Task 1');
    });

    it('should return filtered tasks by status', async () => {
      const filterDto: TaskFilterDto = {
        status: TaskStatus.PENDING,
      };

      const expectedResponse: TaskResponseDto[] = [
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

      mockTasksService.findAll.mockResolvedValue(expectedResponse);

      const result = await controller.findAll(filterDto);

      expect(service.findAll).toHaveBeenCalledWith(filterDto);
      expect(result).toHaveLength(1);
      expect(result[0].status).toBe(TaskStatus.PENDING);
    });

    it('should return filtered tasks by priority', async () => {
      const filterDto: TaskFilterDto = {
        priority: TaskPriority.HIGH,
      };

      const expectedResponse: TaskResponseDto[] = [
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

      mockTasksService.findAll.mockResolvedValue(expectedResponse);

      const result = await controller.findAll(filterDto);

      expect(service.findAll).toHaveBeenCalledWith(filterDto);
      expect(result).toHaveLength(1);
      expect(result[0].priority).toBe(TaskPriority.HIGH);
    });

    it('should return empty array when no tasks match filters', async () => {
      const filterDto: TaskFilterDto = {
        status: TaskStatus.COMPLETED,
        priority: TaskPriority.HIGH,
      };

      mockTasksService.findAll.mockResolvedValue([]);

      const result = await controller.findAll(filterDto);

      expect(service.findAll).toHaveBeenCalledWith(filterDto);
      expect(result).toEqual([]);
    });
  });

  describe('findOne', () => {
    it('should return a single task by ID', async () => {
      const taskId = '1';
      const expectedResponse: TaskResponseDto = {
        id: taskId,
        title: 'Found Task',
        description: 'Task description',
        status: TaskStatus.IN_PROGRESS,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockTasksService.findOne.mockResolvedValue(expectedResponse);

      const result = await controller.findOne(taskId);

      expect(service.findOne).toHaveBeenCalledWith(taskId);
      expect(result).toEqual(expectedResponse);
      expect(result.id).toBe(taskId);
    });

    it('should propagate NotFoundException from service', async () => {
      const taskId = '999';
      const error = new Error('Task not found');

      mockTasksService.findOne.mockRejectedValue(error);

      await expect(controller.findOne(taskId)).rejects.toThrow(error);
      expect(service.findOne).toHaveBeenCalledWith(taskId);
    });
  });

  describe('update', () => {
    it('should update a task successfully', async () => {
      const taskId = '1';
      const updateTaskDto: UpdateTaskDto = {
        title: 'Updated Title',
        status: TaskStatus.COMPLETED,
      };

      const expectedResponse: TaskResponseDto = {
        id: taskId,
        title: 'Updated Title',
        description: 'Original description',
        status: TaskStatus.COMPLETED,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockTasksService.update.mockResolvedValue(expectedResponse);

      const result = await controller.update(taskId, updateTaskDto);

      expect(service.update).toHaveBeenCalledWith(taskId, updateTaskDto);
      expect(result.title).toBe('Updated Title');
      expect(result.status).toBe(TaskStatus.COMPLETED);
    });

    it('should update task with partial data', async () => {
      const taskId = '1';
      const updateTaskDto: UpdateTaskDto = {
        description: 'New description only',
      };

      const expectedResponse: TaskResponseDto = {
        id: taskId,
        title: 'Original Title',
        description: 'New description only',
        status: TaskStatus.PENDING,
        priority: TaskPriority.LOW,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockTasksService.update.mockResolvedValue(expectedResponse);

      const result = await controller.update(taskId, updateTaskDto);

      expect(service.update).toHaveBeenCalledWith(taskId, updateTaskDto);
      expect(result.description).toBe('New description only');
      expect(result.title).toBe('Original Title'); // unchanged
    });

    it('should update task priority', async () => {
      const taskId = '1';
      const updateTaskDto: UpdateTaskDto = {
        priority: TaskPriority.HIGH,
      };

      const expectedResponse: TaskResponseDto = {
        id: taskId,
        title: 'Task Title',
        description: 'Description',
        status: TaskStatus.PENDING,
        priority: TaskPriority.HIGH,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockTasksService.update.mockResolvedValue(expectedResponse);

      const result = await controller.update(taskId, updateTaskDto);

      expect(service.update).toHaveBeenCalledWith(taskId, updateTaskDto);
      expect(result.priority).toBe(TaskPriority.HIGH);
    });

    it('should propagate NotFoundException from service', async () => {
      const taskId = '999';
      const updateTaskDto: UpdateTaskDto = {
        title: 'Updated',
      };
      const error = new Error('Task not found');

      mockTasksService.update.mockRejectedValue(error);

      await expect(controller.update(taskId, updateTaskDto)).rejects.toThrow(error);
      expect(service.update).toHaveBeenCalledWith(taskId, updateTaskDto);
    });
  });

  describe('remove', () => {
    it('should remove a task successfully', async () => {
      const taskId = '1';

      mockTasksService.remove.mockResolvedValue(undefined);

      await controller.remove(taskId);

      expect(service.remove).toHaveBeenCalledWith(taskId);
    });

    it('should resolve void on successful deletion', async () => {
      const taskId = '1';

      mockTasksService.remove.mockResolvedValue(undefined);

      const result = controller.remove(taskId);

      await expect(result).resolves.toBeUndefined();
      expect(service.remove).toHaveBeenCalledWith(taskId);
    });

    it('should propagate NotFoundException from service', async () => {
      const taskId = '999';
      const error = new Error('Task not found');

      mockTasksService.remove.mockRejectedValue(error);

      await expect(controller.remove(taskId)).rejects.toThrow(error);
      expect(service.remove).toHaveBeenCalledWith(taskId);
    });
  });

  describe('Controller Integration', () => {
    it('should handle service errors gracefully', async () => {
      const createTaskDto: CreateTaskDto = {
        title: 'Error Task',
      };
      const error = new Error('Database error');

      mockTasksService.create.mockRejectedValue(error);

      await expect(controller.create(createTaskDto)).rejects.toThrow(error);
    });

    it('should maintain proper request/response flow', async () => {
      const taskId = '1';
      const expectedResponse: TaskResponseDto = {
        id: taskId,
        title: 'Integration Test Task',
        description: 'Testing controller flow',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
        createdAt: new Date('2024-01-01'),
        updatedAt: new Date('2024-01-01'),
      };

      mockTasksService.findOne.mockResolvedValue(expectedResponse);

      const result = await controller.findOne(taskId);

      expect(result).toMatchObject({
        id: taskId,
        title: 'Integration Test Task',
        description: 'Testing controller flow',
        status: TaskStatus.PENDING,
        priority: TaskPriority.MEDIUM,
      });
      expect(result.createdAt).toBeInstanceOf(Date);
      expect(result.updatedAt).toBeInstanceOf(Date);
    });
  });
});