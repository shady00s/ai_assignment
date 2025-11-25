import { Task } from '@/types';
import { useState, useCallback } from 'react';
import { useGetTasksQuery } from '@/store/api/apiSlice';
  
interface UseTaskIntegrationOptions {
  currentTaskId?: string;
  autoRefresh?: boolean;
  filterStatus?: string[];
  priority?: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';
}

interface UseTaskIntegrationReturn {
  tasks: Task[];
  currentTask: Task | undefined;
  isLoading: boolean;
  error: any;
  selectTask: (taskId: string) => void;
  clearCurrentTask: () => void;
  getTaskProgress: (task: Task) => number;
  getNextTask: () => Task | undefined;
  refetch: () => Promise<void>;
}

export const useTaskIntegration = (options: UseTaskIntegrationOptions = {}): UseTaskIntegrationReturn => {
  const {
    currentTaskId,
    autoRefresh = false,
    filterStatus = ['TODO', 'IN_PROGRESS'],
    priority,
  } = options;

  const [selectedTaskId, setSelectedTaskId] = useState<string | undefined>(currentTaskId);

  const {
    data: tasks = [],
    isLoading,
    error,
    refetch,
  } = useGetTasksQuery({
    filters: {
      status: filterStatus,
      ...(priority && { priority: [priority] }),
    },
    sort: {
      field: 'priority',
      direction: 'DESC',
    },
  }, {
    refetchOnMountOrArgChange: true,
    refetchOnWindowFocus: autoRefresh,
  });

  const currentTask = selectedTaskId ? tasks.find(task => task.id === selectedTaskId) : undefined;

  const selectTask = useCallback((taskId: string) => {
    setSelectedTaskId(taskId);
  }, []);

  const clearCurrentTask = useCallback(() => {
    setSelectedTaskId(undefined);
  }, []);

  const getTaskProgress = useCallback((task: Task): number => {
    if (task.estimatedPomodoros === 0) return 0;
    return (task.completedPomodoros / task.estimatedPomodoros) * 100;
  }, []);

  const getNextTask = useCallback((): Task | undefined => {
    // Get tasks that are not completed and have remaining pomodoros
    const availableTasks = tasks.filter(
      task => task.status !== 'COMPLETED' && task.completedPomodoros < task.estimatedPomodoros
    );

    if (availableTasks.length === 0) return undefined;

    // Sort by priority (URGENT > HIGH > MEDIUM > LOW) and then by progress
    availableTasks.sort((a, b) => {
      const priorityOrder = { URGENT: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
      const aPriority = priorityOrder[a.priority];
      const bPriority = priorityOrder[b.priority];

      if (aPriority !== bPriority) {
        return bPriority - aPriority; // Higher priority first
      }

      // If same priority, prefer tasks with less progress
      const aProgress = getTaskProgress(a);
      const bProgress = getTaskProgress(b);

      return aProgress - bProgress;
    });

    return availableTasks[0];
  }, [tasks, getTaskProgress]);

  // Auto-select the next available task if no task is currently selected
  useState(() => {
    if (!selectedTaskId && tasks.length > 0) {
      const nextTask = getNextTask();
      if (nextTask) {
        setSelectedTaskId(nextTask.id);
      }
    }
  });

  return {
    tasks,
    currentTask,
    isLoading,
    error,
    selectTask,
    clearCurrentTask,
    getTaskProgress,
    getNextTask,
    refetch,
  };
};

