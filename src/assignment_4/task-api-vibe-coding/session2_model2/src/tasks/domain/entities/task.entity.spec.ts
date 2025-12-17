import { Task } from './task.entity';
import { TaskId } from '../values/task-id.value';
import { TaskTitle } from '../values/task-title.value';
import { TaskStatusValue } from '../values/task-status.enum';
import { TaskPriorityValue } from '../values/task-priority.enum';
import { TaskCreatedEvent } from '../events/task-created.event';
import { TaskUpdatedEvent } from '../events/task-updated.event';
import { TaskStatusChangedEvent } from '../events/task-status-changed.event';

describe('Task Entity', () => {
  const taskId = TaskId.fromString('test-id');
  const title = TaskTitle.create('Test Task');
  const status = TaskStatusValue.pending();
  const priority = TaskPriorityValue.medium();

  describe('Task Creation', () => {
    it('should create a task with valid properties', () => {
      const task = Task.create(taskId, title, status, priority);

      expect(task.getId()).toEqual(taskId);
      expect(task.getTitle()).toEqual(title);
      expect(task.getStatus()).toEqual(status);
      expect(task.getPriority()).toEqual(priority);
      expect(task.getDescription()).toBeNull();
      expect(task.getCreatedAt()).toBeInstanceOf(Date);
      expect(task.getUpdatedAt()).toBeInstanceOf(Date);
    });

    it('should create a task with description', () => {
      const description = 'Task description';
      const task = Task.create(taskId, title, status, priority, description);

      expect(task.getDescription()).toBe(description);
    });

    it('should create a task with generated ID', () => {
      const task = Task.createWithGeneratedId(title, status, priority);

      expect(task.getId()).toBeInstanceOf(TaskId);
      expect(task.getId().getValue()).toBeTruthy();
    });

    it('should generate TaskCreatedEvent on creation', () => {
      const task = Task.create(taskId, title, status, priority);
      const events = task.getUncommittedEvents();

      expect(events).toHaveLength(1);
      expect(events[0]).toBeInstanceOf(TaskCreatedEvent);
      expect(events[0].taskId).toEqual(taskId);
      expect(events[0].title).toEqual(title.getValue());
      expect(events[0].status).toEqual(status.getValue());
      expect(events[0].priority).toEqual(priority.getValue());
    });
  });

  describe('Task Reconstruction', () => {
    it('should reconstruct a task from existing data', () => {
      const createdAt = new Date('2024-01-01');
      const updatedAt = new Date('2024-01-02');
      const description = 'Existing task';

      const task = Task.reconstruct(
        taskId,
        title,
        status,
        priority,
        description,
        createdAt,
        updatedAt
      );

      expect(task.getId()).toEqual(taskId);
      expect(task.getTitle()).toEqual(title);
      expect(task.getStatus()).toEqual(status);
      expect(task.getPriority()).toEqual(priority);
      expect(task.getDescription()).toBe(description);
      expect(task.getCreatedAt()).toEqual(createdAt);
      expect(task.getUpdatedAt()).toEqual(updatedAt);
      expect(task.getUncommittedEvents()).toHaveLength(0);
    });
  });

  describe('Task Updates', () => {
    it('should update task title', () => {
      const task = Task.create(taskId, title, status, priority);
      task.clearUncommittedEvents();

      const newTitle = TaskTitle.create('Updated Task');
      task.updateTitle(newTitle);

      expect(task.getTitle()).toEqual(newTitle);
      const events = task.getUncommittedEvents();
      expect(events).toHaveLength(1);
      expect(events[0]).toBeInstanceOf(TaskUpdatedEvent);
    });

    it('should update task description', () => {
      const task = Task.create(taskId, title, status, priority);
      task.clearUncommittedEvents();

      const newDescription = 'Updated description';
      task.updateDescription(newDescription);

      expect(task.getDescription()).toBe(newDescription);
      const events = task.getUncommittedEvents();
      expect(events).toHaveLength(1);
      expect(events[0]).toBeInstanceOf(TaskUpdatedEvent);
    });

    it('should update task priority', () => {
      const task = Task.create(taskId, title, status, priority);
      task.clearUncommittedEvents();

      const newPriority = TaskPriorityValue.high();
      task.updatePriority(newPriority);

      expect(task.getPriority()).toEqual(newPriority);
      const events = task.getUncommittedEvents();
      expect(events).toHaveLength(1);
      expect(events[0]).toBeInstanceOf(TaskUpdatedEvent);
    });
  });

  describe('Status Changes', () => {
    it('should change status when valid transition', () => {
      const task = Task.create(taskId, title, status, priority);
      task.clearUncommittedEvents();

      const newStatus = TaskStatusValue.inProgress();
      task.changeStatus(newStatus);

      expect(task.getStatus()).toEqual(newStatus);
      const events = task.getUncommittedEvents();
      expect(events).toHaveLength(1);
      expect(events[0]).toBeInstanceOf(TaskStatusChangedEvent);
    });

    it('should throw error when invalid status transition', () => {
      const task = Task.create(taskId, title, TaskStatusValue.completed(), priority);

      const invalidStatus = TaskStatusValue.pending();

      expect(() => {
        task.changeStatus(invalidStatus);
      }).toThrow('Cannot transition from COMPLETED to PENDING');
    });

    it('should allow same status (no-op)', () => {
      const task = Task.create(taskId, title, status, priority);
      task.clearUncommittedEvents();

      task.changeStatus(status);

      expect(task.getStatus()).toEqual(status);
      expect(task.getUncommittedEvents()).toHaveLength(0);
    });
  });

  describe('Task Status Checks', () => {
    it('should correctly identify pending status', () => {
      const task = Task.create(taskId, title, TaskStatusValue.pending(), priority);
      expect(task.isPending()).toBe(true);
      expect(task.isInProgress()).toBe(false);
      expect(task.isCompleted()).toBe(false);
    });

    it('should correctly identify in-progress status', () => {
      const task = Task.create(taskId, title, TaskStatusValue.inProgress(), priority);
      expect(task.isPending()).toBe(false);
      expect(task.isInProgress()).toBe(true);
      expect(task.isCompleted()).toBe(false);
    });

    it('should correctly identify completed status', () => {
      const task = Task.create(taskId, title, TaskStatusValue.completed(), priority);
      expect(task.isPending()).toBe(false);
      expect(task.isInProgress()).toBe(false);
      expect(task.isCompleted()).toBe(true);
    });
  });

  describe('Event Management', () => {
    it('should clear uncommitted events', () => {
      const task = Task.create(taskId, title, status, priority);
      expect(task.getUncommittedEvents()).toHaveLength(1);

      task.clearUncommittedEvents();
      expect(task.getUncommittedEvents()).toHaveLength(0);
    });
  });

  describe('Task Equality', () => {
    it('should be equal when IDs are same', () => {
      const task1 = Task.create(taskId, title, status, priority);
      const task2 = Task.reconstruct(taskId, title, status, priority);

      expect(task1.equals(task2)).toBe(true);
    });

    it('should not be equal when IDs are different', () => {
      const task1 = Task.create(taskId, title, status, priority);
      const differentId = TaskId.fromString('different-id');
      const task2 = Task.create(differentId, title, status, priority);

      expect(task1.equals(task2)).toBe(false);
    });
  });
});