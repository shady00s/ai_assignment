import { Test, TestingModule } from '@nestjs/testing';
import { CommandBus, EventBus } from '@nestjs/cqrs';
import { CreateTaskHandler } from './create-task.handler';
import { CreateTaskCommand } from '../commands/create-task.command';
import { TaskRepository } from '../../domain/repositories/task.repository.interface';
import { Task } from '../../domain/entities/task.entity';
import { TaskTitle } from '../../domain/values/task-title.value';
import { TaskStatusValue } from '../../domain/values/task-status.enum';
import { TaskPriorityValue } from '../../domain/values/task-priority.enum';
import { TaskId } from '../../domain/values/task-id.value';

describe('CreateTaskHandler', () => {
  let handler: CreateTaskHandler;
  let taskRepository: jest.Mocked<TaskRepository>;
  let eventBus: jest.Mocked<EventBus>;

  beforeEach(async () => {
    const mockTaskRepository = {
      save: jest.fn(),
      findById: jest.fn(),
      findAll: jest.fn(),
      delete: jest.fn(),
      exists: jest.fn(),
    };

    const mockEventBus = {
      publish: jest.fn(),
      publishAll: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        CreateTaskHandler,
        {
          provide: 'TaskRepository',
          useValue: mockTaskRepository,
        },
        {
          provide: EventBus,
          useValue: mockEventBus,
        },
      ],
    }).compile();

    handler = module.get<CreateTaskHandler>(CreateTaskHandler);
    taskRepository = module.get('TaskRepository') as jest.Mocked<TaskRepository>;
    eventBus = module.get(EventBus) as jest.Mocked<EventBus>;
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should create a task with minimal data', async () => {
    // Arrange
    const command = CreateTaskCommand.create('Test Task');
    const expectedTask = Task.createWithGeneratedId(
      TaskTitle.create('Test Task'),
      TaskStatusValue.pending(),
      TaskPriorityValue.default()
    );

    taskRepository.save.mockResolvedValue();

    // Act
    await handler.execute(command);

    // Assert
    expect(taskRepository.save).toHaveBeenCalledWith(expect.any(Task));
    expect(eventBus.publishAll).toHaveBeenCalled();
  });

  it('should create a task with all data', async () => {
    // Arrange
    const command = CreateTaskCommand.create(
      'Complete Task',
      'Task description',
      'HIGH'
    );

    taskRepository.save.mockResolvedValue();

    // Act
    await handler.execute(command);

    // Assert
    expect(taskRepository.save).toHaveBeenCalledWith(expect.any(Task));
    const savedTask = taskRepository.save.mock.calls[0][0] as Task;
    expect(savedTask.getTitle().getValue()).toBe('Complete Task');
    expect(savedTask.getDescription()).toBe('Task description');
    expect(savedTask.getPriority().getValue()).toBe('HIGH');
  });

  it('should publish domain events after creating task', async () => {
    // Arrange
    const command = CreateTaskCommand.create('Test Task');
    taskRepository.save.mockResolvedValue();

    // Act
    await handler.execute(command);

    // Assert
    expect(eventBus.publishAll).toHaveBeenCalled();
    const publishedEvents = eventBus.publishAll.mock.calls[0][0];
    expect(publishedEvents).toHaveLength(1);
    expect(publishedEvents[0].getEventName()).toBe('TaskCreated');
  });

  it('should handle repository errors gracefully', async () => {
    // Arrange
    const command = CreateTaskCommand.create('Test Task');
    const error = new Error('Database error');
    taskRepository.save.mockRejectedValue(error);

    // Act & Assert
    await expect(handler.execute(command)).rejects.toThrow('Database error');
  });
});