import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { Task, TaskStatus, TaskPriority, CreateTaskRequest, UpdateTaskRequest, TaskFilters, TaskSort } from '../../types';

interface TasksState {
  tasks: Task[];
  currentTask: Task | null;
  filters: TaskFilters;
  sort: TaskSort;
  isLoading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: TasksState = {
  tasks: [],
  currentTask: null,
  filters: {},
  sort: {
    field: 'createdAt',
    direction: 'DESC',
  },
  isLoading: false,
  error: null,
  lastUpdated: null,
};

// Async thunks
export const fetchTasks = createAsyncThunk<
  Task[],
  { filters?: TaskFilters; sort?: TaskSort } | undefined,
  { rejectValue: string }
>(
  'tasks/fetchTasks',
  async (params, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: { token: string | null } };
      const token = state.auth.token;

      const queryParams = new URLSearchParams();

      if (params?.filters) {
        Object.entries(params.filters).forEach(([key, value]) => {
          if (value) {
            if (Array.isArray(value)) {
              value.forEach(v => queryParams.append(key, v));
            } else {
              queryParams.append(key, value.toString());
            }
          }
        });
      }

      if (params?.sort) {
        queryParams.set('sortBy', params.sort.field);
        queryParams.set('sortOrder', params.sort.direction);
      }

      const response = await fetch(`/api/tasks?${queryParams.toString()}`, {
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to fetch tasks');
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Failed to fetch tasks');
    }
  }
);

export const createTask = createAsyncThunk<
  Task,
  CreateTaskRequest,
  { rejectValue: string }
>(
  'tasks/createTask',
  async (taskData, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: { token: string | null } };
      const token = state.auth.token;

      const response = await fetch('/api/tasks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': token ? `Bearer ${token}` : '',
        },
        body: JSON.stringify(taskData),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to create task');
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Failed to create task');
    }
  }
);

export const updateTask = createAsyncThunk<
  Task,
  { id: string; updates: UpdateTaskRequest },
  { rejectValue: string }
>(
  'tasks/updateTask',
  async ({ id, updates }, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: { token: string | null } };
      const token = state.auth.token;

      const response = await fetch(`/api/tasks/${id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': token ? `Bearer ${token}` : '',
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to update task');
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Failed to update task');
    }
  }
);

export const deleteTask = createAsyncThunk<
  string,
  string,
  { rejectValue: string }
>(
  'tasks/deleteTask',
  async (taskId, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: { token: string | null } };
      const token = state.auth.token;

      const response = await fetch(`/api/tasks/${taskId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to delete task');
      }

      return taskId;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Failed to delete task');
    }
  }
);

const tasksSlice = createSlice({
  name: 'tasks',
  initialState,
  reducers: {
    setCurrentTask: (state, action: PayloadAction<Task | null>) => {
      state.currentTask = action.payload;
    },
    setFilters: (state, action: PayloadAction<TaskFilters>) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    clearFilters: (state) => {
      state.filters = {};
    },
    setSort: (state, action: PayloadAction<TaskSort>) => {
      state.sort = action.payload;
    },
    updateTaskStatus: (state, action: PayloadAction<{ taskId: string; status: TaskStatus }>) => {
      const task = state.tasks.find(t => t.id === action.payload.taskId);
      if (task) {
        task.status = action.payload.status;
        task.updatedAt = new Date().toISOString();

        if (action.payload.status === 'COMPLETED') {
          task.completedAt = new Date().toISOString();
        }
      }
    },
    incrementTaskProgress: (state, action: PayloadAction<string>) => {
      const task = state.tasks.find(t => t.id === action.payload);
      if (task && task.completedPomodoros < task.estimatedPomodoros) {
        task.completedPomodoros += 1;
        task.updatedAt = new Date().toISOString();

        if (task.completedPomodoros >= task.estimatedPomodoros) {
          task.status = 'COMPLETED';
          task.completedAt = new Date().toISOString();
        }
      }
    },
    decrementTaskProgress: (state, action: PayloadAction<string>) => {
      const task = state.tasks.find(t => t.id === action.payload);
      if (task && task.completedPomodoros > 0) {
        task.completedPomodoros -= 1;
        task.updatedAt = new Date().toISOString();

        if (task.status === 'COMPLETED') {
          task.status = 'IN_PROGRESS';
          task.completedAt = undefined;
        }
      }
    },
    clearError: (state) => {
      state.error = null;
    },
    addLocalTask: (state, action: PayloadAction<Task>) => {
      state.tasks.unshift(action.payload);
      state.lastUpdated = new Date().toISOString();
    },
    updateLocalTask: (state, action: PayloadAction<{ id: string; updates: Partial<Task> }>) => {
      const taskIndex = state.tasks.findIndex(t => t.id === action.payload.id);
      if (taskIndex !== -1) {
        state.tasks[taskIndex] = {
          ...state.tasks[taskIndex],
          ...action.payload.updates,
          updatedAt: new Date().toISOString(),
        };
        state.lastUpdated = new Date().toISOString();
      }
    },
    removeLocalTask: (state, action: PayloadAction<string>) => {
      state.tasks = state.tasks.filter(t => t.id !== action.payload);
      state.lastUpdated = new Date().toISOString();

      if (state.currentTask?.id === action.payload) {
        state.currentTask = null;
      }
    },
  },
  extraReducers: (builder) => {
    // Fetch Tasks
    builder
      .addCase(fetchTasks.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchTasks.fulfilled, (state, action) => {
        state.isLoading = false;
        state.tasks = action.payload;
        state.lastUpdated = new Date().toISOString();
        state.error = null;
      })
      .addCase(fetchTasks.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to fetch tasks';
      });

    // Create Task
    builder
      .addCase(createTask.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(createTask.fulfilled, (state, action) => {
        state.isLoading = false;
        state.tasks.unshift(action.payload);
        state.lastUpdated = new Date().toISOString();
        state.error = null;
      })
      .addCase(createTask.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to create task';
      });

    // Update Task
    builder
      .addCase(updateTask.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(updateTask.fulfilled, (state, action) => {
        state.isLoading = false;
        const taskIndex = state.tasks.findIndex(t => t.id === action.payload.id);
        if (taskIndex !== -1) {
          state.tasks[taskIndex] = action.payload;
          state.lastUpdated = new Date().toISOString();
        }
        state.error = null;
      })
      .addCase(updateTask.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to update task';
      });

    // Delete Task
    builder
      .addCase(deleteTask.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(deleteTask.fulfilled, (state, action) => {
        state.isLoading = false;
        state.tasks = state.tasks.filter(t => t.id !== action.payload);
        state.lastUpdated = new Date().toISOString();

        if (state.currentTask?.id === action.payload) {
          state.currentTask = null;
        }
        state.error = null;
      })
      .addCase(deleteTask.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to delete task';
      });
  },
});

export const {
  setCurrentTask,
  setFilters,
  clearFilters,
  setSort,
  updateTaskStatus,
  incrementTaskProgress,
  decrementTaskProgress,
  clearError,
  addLocalTask,
  updateLocalTask,
  removeLocalTask,
} = tasksSlice.actions;

// Selectors
export const tasksSelectors = {
  selectAllTasks: (state: { tasks: TasksState }) => state.tasks.tasks,
  selectTasksByStatus: (state: { tasks: TasksState }, status: TaskStatus) =>
    state.tasks.tasks.filter(task => task.status === status),
  selectTasksByPriority: (state: { tasks: TasksState }, priority: TaskPriority) =>
    state.tasks.tasks.filter(task => task.priority === priority),
  selectCurrentTask: (state: { tasks: TasksState }) => state.tasks.currentTask,
  selectTaskById: (state: { tasks: TasksState }, taskId: string) =>
    state.tasks.tasks.find(task => task.id === taskId),
  selectFilteredTasks: (state: { tasks: TasksState }) => {
    let filteredTasks = [...state.tasks.tasks];

    // Apply filters
    const { filters } = state.tasks;

    if (filters.status && filters.status.length > 0) {
      filteredTasks = filteredTasks.filter(task =>
        filters.status!.includes(task.status)
      );
    }

    if (filters.priority && filters.priority.length > 0) {
      filteredTasks = filteredTasks.filter(task =>
        filters.priority!.includes(task.priority)
      );
    }

    if (filters.assigneeId && filters.assigneeId.length > 0) {
      filteredTasks = filteredTasks.filter(task =>
        task.assigneeId && filters.assigneeId!.includes(task.assigneeId)
      );
    }

    if (filters.tags && filters.tags.length > 0) {
      filteredTasks = filteredTasks.filter(task =>
        filters.tags!.some(tag => task.tags.includes(tag))
      );
    }

    if (filters.dueDateRange) {
      const { start, end } = filters.dueDateRange;
      filteredTasks = filteredTasks.filter(task => {
        if (!task.dueDate) return false;
        const taskDueDate = new Date(task.dueDate);
        const startDate = new Date(start);
        const endDate = new Date(end);
        return taskDueDate >= startDate && taskDueDate <= endDate;
      });
    }

    // Apply sorting
    const { sort } = state.tasks;
    filteredTasks.sort((a, b) => {
      let aValue: any = a[sort.field];
      let bValue: any = b[sort.field];

      // Handle date fields
      if (sort.field === 'createdAt' || sort.field === 'updatedAt' || sort.field === 'dueDate') {
        aValue = aValue ? new Date(aValue).getTime() : 0;
        bValue = bValue ? new Date(bValue).getTime() : 0;
      }

      // Handle priority
      if (sort.field === 'priority') {
        const priorityOrder = { 'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1 };
        aValue = priorityOrder[aValue as TaskPriority];
        bValue = priorityOrder[bValue as TaskPriority];
      }

      // Handle status
      if (sort.field === 'status') {
        const statusOrder = { 'TODO': 1, 'IN_PROGRESS': 2, 'COMPLETED': 3, 'CANCELLED': 4 };
        aValue = statusOrder[aValue as TaskStatus];
        bValue = statusOrder[bValue as TaskStatus];
      }

      if (aValue < bValue) return sort.direction === 'ASC' ? -1 : 1;
      if (aValue > bValue) return sort.direction === 'ASC' ? 1 : -1;
      return 0;
    });

    return filteredTasks;
  },
  selectTasksLoading: (state: { tasks: TasksState }) => state.tasks.isLoading,
  selectTasksError: (state: { tasks: TasksState }) => state.tasks.error,
  selectTasksFilters: (state: { tasks: TasksState }) => state.tasks.filters,
  selectTasksSort: (state: { tasks: TasksState }) => state.tasks.sort,
};

export { tasksSlice };
export type { TasksState };