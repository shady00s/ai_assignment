import React, { useState, useMemo, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { useAppSelector } from '../../../hooks/redux';
import { useGetTasksQuery, useCreateTaskMutation, useUpdateTaskMutation, useDeleteTaskMutation } from '../../../store/api/apiSlice';
import { Task, CreateTaskRequest, UpdateTaskRequest } from '../../../types';
import {
  DndContext,
  DragEndEvent,
  DragOverEvent,
  DragStartEvent,
  PointerSensor,
  useSensor,
  useSensors,
  DragOverlay,
  closestCorners,
} from '@dnd-kit/core';
import { KanbanColumn } from '../../molecules/KanbanColumn';
import { TaskCard } from '../../molecules/TaskCard';
import { Button } from '../../atoms/Button';

const TaskBoardContainer = styled.div`
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  width: 100%;
  max-width: 100vw;
  overflow-x: hidden;

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme }) => theme.spacing.lg};
    max-width: 100%;
  }
`;

const Header = styled.header`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.lg};
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    margin-bottom: ${({ theme }) => theme.spacing.tablet.xl};
    gap: ${({ theme }) => theme.spacing.tablet.md};
    align-items: center;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    margin-bottom: 32px;
    gap: 16px;
    align-items: center;
  }
`;

const HeaderContent = styled.div`
  flex: 1;
`;

const Title = styled.h1`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: #2C3E50;
  margin: 0;
  font-family: ${({ theme }) => theme.typography.fontFamily.secondary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet['3xl']};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 2rem;
  }
`;

const Subtitle = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.base};
  color: #8B7D7B;
  margin: ${({ theme }) => theme.spacing.mobile.xs} 0 0 0;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.base};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 1rem;
  }
`;

const AddTaskButton = styled(Button)`
  padding: ${({ theme }) => theme.spacing.mobile.sm} ${({ theme }) => theme.spacing.mobile.md};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  white-space: nowrap;
  flex-shrink: 0;

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.sm} ${({ theme }) => theme.spacing.tablet.lg};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 12px 24px;
    font-size: 14px;
    gap: 8px;
  }
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.lg};

  ${({ theme }) => theme.mediaQueries.tablet} {
    grid-template-columns: repeat(4, 1fr);
    gap: ${({ theme }) => theme.spacing.tablet.md};
    margin-bottom: ${({ theme }) => theme.spacing.tablet.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
  }
`;

const StatCard = styled.div<{ $gradient: string }>`
  background: ${({ $gradient }) => $gradient};
  color: white;
  padding: ${({ theme }) => theme.spacing.mobile.lg};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  text-align: center;

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.lg};
    border-radius: ${({ theme }) => theme.borderRadius.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 20px;
    border-radius: 12px;
  }
`;

const StatValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xs};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 24px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 24px;
  }
`;

const StatLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  opacity: 0.9;

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 14px;
  }
`;

const KanbanBoard = styled.div`
  background: linear-gradient(135deg, #F8F9FA 0%, #E8D8C8 100%);
  padding: ${({ theme }) => theme.spacing.mobile.lg};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 1px solid rgba(127, 168, 112, 0.1);
  position: relative;
  overflow: hidden;

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.xl};
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 24px;
    border-radius: 16px;
  }
`;

const KanbanColumns = styled.div`
  display: flex;
  gap: 12px;
  padding: 0 ${({ theme }) => theme.spacing.mobile.sm} 16px ${({ theme }) => theme.spacing.mobile.sm};
  width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  -webkit-overflow-scrolling: touch;
  scroll-snap-type: x mandatory;
  scroll-behavior: smooth;
  position: relative;

  /* Hide scrollbar for mobile */
  -ms-overflow-style: none;
  scrollbar-width: none;
  &::-webkit-scrollbar {
    display: none;
  }

  ${({ theme }) => theme.mediaQueries.mobile} {
    gap: 8px;
    padding: 0 ${({ theme }) => theme.spacing.mobile.xs} 12px ${({ theme }) => theme.spacing.mobile.xs};
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: 12px;
    padding: 0 ${({ theme }) => theme.spacing.tablet.sm} 16px ${({ theme }) => theme.spacing.tablet.sm};
    scroll-snap-type: x proximity;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 20px;
    padding: 0;
    overflow-x: visible;
    justify-content: center;
    scroll-snap-type: none;

    /* Show scrollbar for desktop */
    &::-webkit-scrollbar {
      display: block;
      height: 8px;
    }

    &::-webkit-scrollbar-track {
      background: #f1f1f1;
      border-radius: 4px;
    }

    &::-webkit-scrollbar-thumb {
      background: #c1c1c1;
      border-radius: 4px;
    }

    &::-webkit-scrollbar-thumb:hover {
      background: #a8a8a8;
    }
  }
`;

const SwipeIndicator = styled.div<{ $direction: 'left' | 'right' }>`
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  ${({ $direction }) => $direction === 'left' ? 'left: 8px;' : 'right: 8px;'}
  background: rgba(127, 168, 112, 0.9);
  color: white;
  border-radius: 50%;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  z-index: 10;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  opacity: 0;
  transition: opacity 0.3s ease;
  pointer-events: none;

  &.visible {
    opacity: 1;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    display: none;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    display: none;
  }
`;

const ColumnIndicator = styled.div<{ $active: boolean; $total: number; $current: number }>`
  position: absolute;
  bottom: 8px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 6px;
  z-index: 10;

  ${({ theme }) => theme.mediaQueries.tablet} {
    display: none;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    display: none;
  }
`;

const IndicatorDot = styled.div<{ $active: boolean }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${({ $active }) => $active ? '#7FA870' : '#D4C5B9'};
  transition: all 0.3s ease;

  ${({ $active }) => $active && `
    transform: scale(1.2);
    box-shadow: 0 2px 4px rgba(127, 168, 112, 0.4);
  `}
`;

const MobileNavigationHint = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: ${({ theme }) => theme.spacing.mobile.md};
  padding: ${({ theme }) => theme.spacing.mobile.sm} ${({ theme }) => theme.spacing.mobile.md};
  background: rgba(212, 197, 185, 0.2);
  border-radius: 20px;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #8B7D7B;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  ${({ theme }) => theme.mediaQueries.tablet} {
    display: none;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    display: none;
  }
`;

const FooterInstructions = styled.div`
  text-align: center;
  margin-top: ${({ theme }) => theme.spacing.mobile.xl};
  padding: ${({ theme }) => theme.spacing.mobile.md};
  background-color: #F8F9FA;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  color: #8B7D7B;

  ${({ theme }) => theme.mediaQueries.tablet} {
    margin-top: ${({ theme }) => theme.spacing.tablet.xl};
    padding: ${({ theme }) => theme.spacing.tablet.md};
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.sm};
    border-radius: ${({ theme }) => theme.borderRadius.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    margin-top: 24px;
    padding: 16px;
    font-size: 14px;
    border-radius: 8px;
  }
`;

interface TaskBoardScreenProps {
  className?: string;
}

export const TaskBoardScreen: React.FC<TaskBoardScreenProps> = ({ className }) => {
  // RTK Query hooks
  const {
    data: tasks = [],
    isLoading,
    error,
    refetch
  } = useGetTasksQuery();

  const [createTask] = useCreateTaskMutation();
  const [updateTask] = useUpdateTaskMutation();
  const [deleteTask] = useDeleteTaskMutation();

  const [activeTask, setActiveTask] = useState<Task | null>(null);

  // Mobile navigation state
  const [currentColumnIndex, setCurrentColumnIndex] = useState(0);
  const [showLeftIndicator, setShowLeftIndicator] = useState(false);
  const [showRightIndicator, setShowRightIndicator] = useState(false);
  const columnsRef = useRef<HTMLDivElement>(null);
  const totalColumns = 3; // TODO, IN_PROGRESS, COMPLETED

  // Mobile scroll detection and indicators
  useEffect(() => {
    const handleScroll = () => {
      if (!columnsRef.current) return;

      const scrollLeft = columnsRef.current.scrollLeft;
      const scrollWidth = columnsRef.current.scrollWidth;
      const clientWidth = columnsRef.current.clientWidth;

      // Get actual column width by measuring the first column
      const firstColumn = columnsRef.current.querySelector('[data-column-id]') as HTMLElement;
      let actualColumnWidth = clientWidth; // Fallback

      if (firstColumn) {
         const columnWidth = firstColumn.offsetWidth;
        const gap = 12; // Match the gap in KanbanColumns styled component
        actualColumnWidth = columnWidth + gap;
      }

      const newColumnIndex = Math.round(scrollLeft / actualColumnWidth);
      setCurrentColumnIndex(Math.max(0, Math.min(newColumnIndex, totalColumns - 1)));

      // Show/hide navigation indicators
      const threshold = 50;
      setShowLeftIndicator(scrollLeft > threshold);
      setShowRightIndicator(scrollLeft < scrollWidth - clientWidth - threshold);
    };

    const columnsElement = columnsRef.current;
    if (columnsElement) {
      columnsElement.addEventListener('scroll', handleScroll, { passive: true });

      // Initial check after a brief delay to allow DOM to render
      setTimeout(handleScroll, 100);

      return () => {
        columnsElement.removeEventListener('scroll', handleScroll);
      };
    }
  }, [totalColumns]);

  // Auto-scroll to column on mobile (helpful after drag operations)
  const scrollToColumn = (columnIndex: number) => {
    if (!columnsRef.current) return;

    // Get actual column width by measuring the first column
    const firstColumn = columnsRef.current.querySelector('[data-column-id]') as HTMLElement;
    let actualColumnWidth = columnsRef.current.clientWidth; // Fallback

    if (firstColumn) {
      const columnWidth = firstColumn.offsetWidth;
      const gap = 12; // Match the gap in KanbanColumns styled component
      actualColumnWidth = columnWidth + gap;
    }

    const targetScrollLeft = columnIndex * actualColumnWidth;

    columnsRef.current.scrollTo({
      left: targetScrollLeft,
      behavior: 'smooth'
    });
  };

  // Configure drag sensors with mobile optimizations
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8, // Minimum distance before drag starts
      },
    })
  );

  // Group tasks by status
  const tasksByStatus = useMemo(() => {
    const grouped = {
      TODO: tasks.filter(task => task.status === 'TODO'),
      IN_PROGRESS: tasks.filter(task => task.status === 'IN_PROGRESS'),
      COMPLETED: tasks.filter(task => task.status === 'COMPLETED'),
    };
    return grouped;
  }, [tasks]);

  // Handle drag start
  const handleDragStart = (event: DragStartEvent) => {
    const { active } = event;
    const task = tasks.find(t => t.id === active.id);
    setActiveTask(task || null);
  };

  // Handle drag over - local visual updates only, Redux updates happen on drag end
  const handleDragOver = (event: DragOverEvent) => {
    const { active, over } = event;
    if (!over) return;

    const activeTask = tasks.find(t => t.id === active.id);

    if (!activeTask) return;

    // Visual reordering only - actual updates will be handled by Redux on drag end
    // This is just for immediate visual feedback during dragging
  };

  // Handle drag end
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (!over) {
      setActiveTask(null);
      return;
    }

    const activeTask = tasks.find(t => t.id === active.id);
    if (!activeTask) {
      setActiveTask(null);
      return;
    }

    let columnChanged = false;
    let targetColumn: Task['status'] = activeTask.status;

    // If dropping on a column
    if (over.id === 'TODO' || over.id === 'IN_PROGRESS' || over.id === 'COMPLETED') {
      if (activeTask.status !== over.id) {
        columnChanged = true;
        targetColumn = over.id as Task['status'];
        // Update via RTK Query
        updateTask({
          id: activeTask.id,
          updates: { status: targetColumn }
        }).unwrap()
          .then(() => {
            // Refetch to ensure UI is updated
            refetch();
          })
          .catch((error) => {
            console.error('Failed to update task:', error);
          });
      }
    }
    // If dropping on another task
    else if (over.data?.current?.type === 'TaskCard') {
      const overTask = tasks.find(t => t.id === over.id);
      if (overTask && activeTask.status !== overTask.status) {
        columnChanged = true;
        targetColumn = overTask.status;
        // Update via RTK Query
        updateTask({
          id: activeTask.id,
          updates: { status: targetColumn }
        }).unwrap()
          .then(() => {
            // Refetch to ensure UI is updated
            refetch();
          })
          .catch((error) => {
            console.error('Failed to update task:', error);
          });
      }
    }

    // Mobile enhancement: Auto-scroll to the column where the task was moved
    if (columnChanged && typeof window !== 'undefined') {
      const isMobile = window.innerWidth < 768;
      if (isMobile) {
        const columnIndex = targetColumn === 'TODO' ? 0 : targetColumn === 'IN_PROGRESS' ? 1 : 2;
        setTimeout(() => scrollToColumn(columnIndex), 300); // Small delay to allow DOM update
      }
    }

    setActiveTask(null);
  };

  // Add new task - creates a temporary local card for editing
  const handleAddTask = () => {
    const tempId = `temp-${Date.now()}`;
    const tempTask: Task = {
      id: tempId,
      title: 'New Task',
      description: 'Click to edit this task',
      priority: 'MEDIUM',
      status: 'TODO',
      estimatedPomodoros: 1,
      completedPomodoros: 0,
      tags: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    // This would require a more complex Redux setup to handle temporary tasks
    // For now, let's create the task with only the fields the backend accepts
    const taskDataForBackend = {
      title: tempTask.title,
      description: tempTask.description,
      priority: tempTask.priority,
      estimatedPomodoros: tempTask.estimatedPomodoros,
      tags: tempTask.tags,
    };

    createTask(taskDataForBackend).unwrap()
      .then(() => {
        // Refetch to ensure UI is updated
        refetch();
      })
      .catch((error) => {
        console.error('Failed to create task:', error);
      });
  };

  // Update task
  const handleUpdateTask = (taskId: string, updates: Partial<Task>) => {
    updateTask({
      id: taskId,
      updates
    }).unwrap()
      .then(() => {
        // Refetch to ensure UI is updated
        refetch();
      })
      .catch((error) => {
        console.error('Failed to update task:', error);
      });
  };

  // Delete task
  const handleDeleteTask = (taskId: string) => {
    deleteTask(taskId).unwrap()
      .then(() => {
        // Refetch to ensure UI is updated
        refetch();
      })
      .catch((error) => {
        console.error('Failed to delete task:', error);
      });
  };

  // Get task statistics
  const stats = useMemo(() => {
    const totalTasks = tasks.length;
    const completedTasks = tasks.filter(t => t.status === 'COMPLETED').length;
    const inProgressTasks = tasks.filter(t => t.status === 'IN_PROGRESS').length;
    const todoTasks = tasks.filter(t => t.status === 'TODO').length;

    const totalEstimated = tasks.reduce((sum, task) => sum + (task.estimatedPomodoros || 0), 0);
    const totalCompleted = tasks.reduce((sum, task) => sum + (task.completedPomodoros || 0), 0);

    return {
      totalTasks,
      completedTasks,
      inProgressTasks,
      todoTasks,
      totalEstimated,
      totalCompleted,
      completionRate: totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0,
    };
  }, [tasks]);

  return (
    <TaskBoardContainer className={className}>
      {/* Header */}
      <Header>
        <HeaderContent>
          <Title>📋 Task Board</Title>
          <Subtitle>Manage your tasks with drag-and-drop simplicity</Subtitle>
        </HeaderContent>

        <AddTaskButton onClick={handleAddTask} disabled={isLoading}>
          <span>➕</span>
          {isLoading ? 'Loading...' : 'Add Task'}
        </AddTaskButton>
      </Header>

      {/* Loading State */}
      {isLoading && tasks.length === 0 && (
        <div style={{
          textAlign: 'center',
          padding: '60px 20px',
          color: '#8B7D7B'
        }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>📋</div>
          <h2 style={{ color: '#2C3E50', marginBottom: '8px' }}>Loading Tasks...</h2>
          <p>Please wait while we fetch your tasks from the server</p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div style={{
          backgroundColor: '#fee',
          color: '#c00',
          padding: '16px',
          borderRadius: '8px',
          marginBottom: '24px',
          border: '1px solid #fcc'
        }}>
          <strong>Error:</strong> {typeof error === 'string' ? error : error?.data?.message || 'Failed to load tasks'}
        </div>
      )}

      {/* Statistics and Board - only show when not loading for first time */}
      {!isLoading || tasks.length > 0 ? (
        <>
          {/* Statistics */}
          <StatsGrid>
        <StatCard $gradient="linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%)">
          <StatValue>{stats.totalTasks}</StatValue>
          <StatLabel>Total Tasks</StatLabel>
        </StatCard>

        <StatCard $gradient="linear-gradient(135deg, #F4A261 0%, #F5B789 100%)">
          <StatValue>{stats.inProgressTasks}</StatValue>
          <StatLabel>In Progress</StatLabel>
        </StatCard>

        <StatCard $gradient="linear-gradient(135deg, #7FA870 0%, #8FBC8F 100%)">
          <StatValue>{stats.completedTasks}</StatValue>
          <StatLabel>Completed</StatLabel>
        </StatCard>

        <StatCard $gradient="linear-gradient(135deg, #E9C46A 0%, #EED989 100%)">
          <StatValue>{stats.totalCompleted}/{stats.totalEstimated}</StatValue>
          <StatLabel>Pomodoros 🍅</StatLabel>
        </StatCard>
      </StatsGrid>

      {/* Kanban Board */}
      <KanbanBoard role="main" aria-label="Task Board">
        <DndContext
          sensors={sensors}
          collisionDetection={closestCorners}
          onDragStart={handleDragStart}
          onDragOver={handleDragOver}
          onDragEnd={handleDragEnd}
        >
          <KanbanColumns
            ref={columnsRef}
            role="region"
            aria-label="Task columns"
            aria-describedby="board-instructions"
          >
            {/* Mobile Navigation Indicators */}
            <SwipeIndicator
              $direction="left"
              className={showLeftIndicator ? 'visible' : ''}
              role="button"
              aria-label="Scroll to previous column"
              tabIndex={-1}
            >
              ◀
            </SwipeIndicator>
            <SwipeIndicator
              $direction="right"
              className={showRightIndicator ? 'visible' : ''}
              role="button"
              aria-label="Scroll to next column"
              tabIndex={-1}
            >
              ▶
            </SwipeIndicator>

            <KanbanColumn
              id="TODO"
              title="To Do"
              tasks={tasksByStatus.TODO}
              onUpdateTask={handleUpdateTask}
              onDeleteTask={handleDeleteTask}
            />
            <KanbanColumn
              id="IN_PROGRESS"
              title="In Progress"
              tasks={tasksByStatus.IN_PROGRESS}
              onUpdateTask={handleUpdateTask}
              onDeleteTask={handleDeleteTask}
            />
            <KanbanColumn
              id="COMPLETED"
              title="Completed"
              tasks={tasksByStatus.COMPLETED}
              onUpdateTask={handleUpdateTask}
              onDeleteTask={handleDeleteTask}
            />

            {/* Mobile Column Position Indicator */}
            <ColumnIndicator
              $total={totalColumns}
              $current={currentColumnIndex}
              $active
              role="progressbar"
              aria-label={`Column ${currentColumnIndex + 1} of ${totalColumns}`}
              aria-valuemin={1}
              aria-valuemax={totalColumns}
              aria-valuenow={currentColumnIndex + 1}
            >
              {[...Array(totalColumns)].map((_, index) => (
                <IndicatorDot
                  key={index}
                  $active={index === currentColumnIndex}
                  aria-label={index === currentColumnIndex ? 'Current column' : `Column ${index + 1}`}
                />
              ))}
            </ColumnIndicator>
          </KanbanColumns>

          <DragOverlay>
            {activeTask ? (
              <div
                style={{ opacity: 0.8, transform: 'rotate(5deg)' }}
                role="tooltip"
                aria-label={`Dragging ${activeTask.title}`}
              >
                <TaskCard {...activeTask} />
              </div>
            ) : null}
          </DragOverlay>
        </DndContext>

        {/* Mobile Navigation Hint */}
        <MobileNavigationHint id="board-instructions">
          👈 Swipe to navigate columns • Drag cards to move them 👉
        </MobileNavigationHint>
      </KanbanBoard>

      {/* Footer Instructions */}
      <FooterInstructions>
        💡 <strong>Tip:</strong> Drag tasks between columns to update their status. Click and hold to start dragging.
      </FooterInstructions>
        </>
      ) : null}
    </TaskBoardContainer>
  );
};

export type { TaskBoardScreenProps };