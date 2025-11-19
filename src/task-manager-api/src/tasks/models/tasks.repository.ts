import { Injectable } from '@nestjs/common';
import * as fs from 'fs/promises';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { Task, TaskStatus } from '../entities/task.entity';
import { TaskData } from './tasks.model';

@Injectable()
export class TasksRepository {
  private readonly filePath = path.join(process.cwd(), 'data', 'tasks.json');
  private tasks: Task[] = [];

  constructor() {
    this.loadTasks();
  }

  private async loadTasks(): Promise<void> {
    try {
      const data = await fs.readFile(this.filePath, 'utf-8');
      const taskData: TaskData = JSON.parse(data);
      this.tasks = taskData.tasks.map(task => ({
        ...task,
        status: task.status as TaskStatus,
        created_at: new Date(task.created_at),
        updated_at: new Date(task.updated_at),
      }));
    } catch (error) {
      this.tasks = [];
      await this.saveTasks();
    }
  }

  private async saveTasks(): Promise<void> {
    const taskData: TaskData = {
      tasks: this.tasks.map(task => ({
        ...task,
        created_at: task.created_at.toISOString(),
        updated_at: task.updated_at.toISOString(),
      })),
      metadata: {
        total_tasks: this.tasks.length,
        last_updated: new Date().toISOString(),
        version: '1.0.0',
      },
    };

    await fs.writeFile(this.filePath, JSON.stringify(taskData, null, 2));
  }

  async findAll(): Promise<Task[]> {
    return [...this.tasks];
  }

  async findById(id: string): Promise<Task | null> {
    return this.tasks.find(task => task.id === id) || null;
  }

  async save(task: Task): Promise<Task> {
    const newTask = {
      ...task,
      id: task.id || uuidv4(),
      created_at: task.created_at || new Date(),
      updated_at: new Date(),
    };

    this.tasks.push(newTask);
    await this.saveTasks();
    return newTask;
  }

  async update(id: string, updates: Partial<Task>): Promise<Task | null> {
    const taskIndex = this.tasks.findIndex(task => task.id === id);
    if (taskIndex === -1) {
      return null;
    }

    this.tasks[taskIndex] = {
      ...this.tasks[taskIndex],
      ...updates,
      updated_at: new Date(),
    };

    await this.saveTasks();
    return this.tasks[taskIndex];
  }

  async delete(id: string): Promise<boolean> {
    const taskIndex = this.tasks.findIndex(task => task.id === id);
    if (taskIndex === -1) {
      return false;
    }

    this.tasks.splice(taskIndex, 1);
    await this.saveTasks();
    return true;
  }

  async findByStatus(status: TaskStatus): Promise<Task[]> {
    return this.tasks.filter(task => task.status === status);
  }

  async searchTasks(query: string): Promise<Task[]> {
    const lowerQuery = query.toLowerCase();
    return this.tasks.filter(task =>
      task.title.toLowerCase().includes(lowerQuery) ||
      (task.description && task.description.toLowerCase().includes(lowerQuery))
    );
  }
}