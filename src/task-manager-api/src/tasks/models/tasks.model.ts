export interface TaskData {
  tasks: Array<{
    id: string;
    title: string;
    description?: string;
    status: 'pending' | 'completed';
    created_at: string;
    updated_at: string;
  }>;
  metadata: {
    total_tasks: number;
    last_updated: string;
    version: string;
  };
}