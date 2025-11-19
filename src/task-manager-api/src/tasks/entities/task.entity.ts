export enum TaskStatus {
  PENDING = 'pending',
  COMPLETED = 'completed',
}

export class Task {
  id: string;
  title: string;
  description?: string;
  status: TaskStatus;
  created_at: Date;
  updated_at: Date;
}