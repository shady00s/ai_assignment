import { Test, TestingModule } from '@nestjs/testing';
import { CqrsTaskController } from './cqrs-task.controller';
import { CommandBus, QueryBus } from '@nestjs/cqrs';
import { TaskResponseDto } from './application/dtos/task-response.dto';
import { CreateTaskCommand } from './application/commands/create-task.command';
import { UpdateTaskCommand } from './application/commands/update-task.command';
import { DeleteTaskCommand } from './application/commands/delete-task.command';
import { GetTaskByIdQuery } from './application/queries/get-task-by-id.query';
import { GetTasksQuery } from './application/queries/get-tasks.query';

describe('CqrsTaskController', () => {
  let controller: CqrsTaskController;
  let commandBus: jest.Mocked<CommandBus>;
  let queryBus: jest.Mocked<QueryBus>;

  const mockTaskResponse: TaskResponseDto = {
    id: 'test-id',
    title: 'Test Task',
    description: 'Test description',
    status: 'PENDING',
    priority: 'MEDIUM',
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  beforeEach(async () => {
    const mockCommandBus = {
      execute: jest.fn(),
    };

    const mockQueryBus = {
      execute: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [CqrsTaskController],
      providers: [
        {
          provide: CommandBus,
          useValue: mockCommandBus,
        },
        {
          provide: QueryBus,
          useValue: mockQueryBus,
        },
      ],
    }).compile();

    controller = module.get<CqrsTaskController>(CqrsTaskController);
    commandBus = module.get(CommandBus) as jest.Mocked<CommandBus>;
    queryBus = module.get(QueryBus) as jest.Mocked<QueryBus>;
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('create', () => {
    it('should create a task using CQRS pattern', async () => {
      // Arrange
      const createDto = {
        title: 'New Task',
        description: 'Task description',
        priority: 'HIGH',
      };

      commandBus.execute.mockResolvedValue(undefined);
      queryBus.execute.mockResolvedValue([mockTaskResponse]);

      // Act
      const result = await controller.create(createDto);

      // Assert
      expect(commandBus.execute).toHaveBeenCalledWith(
        expect.any(CreateTaskCommand)
      );
      expect(queryBus.execute).toHaveBeenCalledWith(expect.any(GetTasksQuery));
      expect(result).toEqual(mockTaskResponse);
    });

    it('should create a task with minimal data', async () => {
      // Arrange
      const createDto = { title: 'Simple Task' };

      commandBus.execute.mockResolvedValue(undefined);
      queryBus.execute.mockResolvedValue([mockTaskResponse]);

      // Act
      const result = await controller.create(createDto);

      // Assert
      expect(commandBus.execute).toHaveBeenCalled();
      expect(result).toEqual(mockTaskResponse);
    });
  });

  describe('findAll', () => {
    it('should return all tasks using CQRS query', async () => {
      // Arrange
      const filterDto = { status: 'PENDING', priority: 'HIGH' };
      const expectedTasks = [mockTaskResponse];

      queryBus.execute.mockResolvedValue(expectedTasks);

      // Act
      const result = await controller.findAll(filterDto);

      // Assert
      expect(queryBus.execute).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'PENDING',
          priority: 'HIGH',
        })
      );
      expect(result).toEqual(expectedTasks);
    });

    it('should return all tasks without filters', async () => {
      // Arrange
      queryBus.execute.mockResolvedValue([mockTaskResponse]);

      // Act
      const result = await controller.findAll({});

      // Assert
      expect(queryBus.execute).toHaveBeenCalledWith(expect.any(GetTasksQuery));
      expect(result).toEqual([mockTaskResponse]);
    });
  });

  describe('findOne', () => {
    it('should return a task by ID using CQRS query', async () => {
      // Arrange
      const taskId = 'test-id';
      queryBus.execute.mockResolvedValue(mockTaskResponse);

      // Act
      const result = await controller.findOne(taskId);

      // Assert
      expect(queryBus.execute).toHaveBeenCalledWith(
        expect.objectContaining({
          id: taskId,
        })
      );
      expect(result).toEqual(mockTaskResponse);
    });
  });

  describe('update', () => {
    it('should update a task using CQRS command', async () => {
      // Arrange
      const taskId = 'test-id';
      const updateDto = {
        title: 'Updated Task',
        status: 'IN_PROGRESS',
      };

      commandBus.execute.mockResolvedValue(undefined);
      queryBus.execute.mockResolvedValue(mockTaskResponse);

      // Act
      const result = await controller.update(taskId, updateDto);

      // Assert
      expect(commandBus.execute).toHaveBeenCalledWith(
        expect.objectContaining({
          id: taskId,
          title: 'Updated Task',
          status: 'IN_PROGRESS',
        })
      );
      expect(queryBus.execute).toHaveBeenCalledWith(
        expect.objectContaining({
          id: taskId,
        })
      );
      expect(result).toEqual(mockTaskResponse);
    });
  });

  describe('remove', () => {
    it('should delete a task using CQRS command', async () => {
      // Arrange
      const taskId = 'test-id';
      commandBus.execute.mockResolvedValue(undefined);

      // Act
      await controller.remove(taskId);

      // Assert
      expect(commandBus.execute).toHaveBeenCalledWith(
        expect.objectContaining({
          id: taskId,
        })
      );
    });
  });

  describe('CQRS Pattern Validation', () => {
    it('should use CommandBus for write operations', async () => {
      // Arrange
      commandBus.execute.mockResolvedValue(undefined);
      queryBus.execute.mockResolvedValue([mockTaskResponse]);

      // Act
      await controller.create({ title: 'Test' });

      // Assert
      expect(commandBus.execute).toHaveBeenCalledWith(
        expect.any(CreateTaskCommand)
      );
    });

    it('should use QueryBus for read operations', async () => {
      // Arrange
      queryBus.execute.mockResolvedValue([mockTaskResponse]);

      // Act
      await controller.findAll({});

      // Assert
      expect(queryBus.execute).toHaveBeenCalledWith(expect.any(GetTasksQuery));
    });

    it('should separate command and query responsibilities', async () => {
      // Test that create (command) uses CommandBus
      commandBus.execute.mockResolvedValue(undefined);
      queryBus.execute.mockResolvedValue([mockTaskResponse]);
      await controller.create({ title: 'Test' });

      // Test that find (query) uses QueryBus
      queryBus.execute.mockResolvedValue([]);
      await controller.findAll({});

      expect(commandBus.execute).toHaveBeenCalledTimes(1);
      expect(queryBus.execute).toHaveBeenCalledTimes(2);
    });
  });

  describe('Error Handling', () => {
    it('should handle command execution errors', async () => {
      // Arrange
      const error = new Error('Command execution failed');
      commandBus.execute.mockRejectedValue(error);

      // Act & Assert
      await expect(
        controller.update('test-id', { title: 'Updated' })
      ).rejects.toThrow('Command execution failed');
    });

    it('should handle query execution errors', async () => {
      // Arrange
      const error = new Error('Query execution failed');
      queryBus.execute.mockRejectedValue(error);

      // Act & Assert
      await expect(controller.findOne('test-id')).rejects.toThrow(
        'Query execution failed'
      );
    });
  });
});