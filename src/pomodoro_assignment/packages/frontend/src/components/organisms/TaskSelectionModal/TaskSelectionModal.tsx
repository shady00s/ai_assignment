import React, { useMemo } from 'react';
import styled from 'styled-components';
import { Task } from '@/types';
import { TaskCard } from '@/components/molecules/TaskCard';
import { Button } from '@/components/atoms/Button';

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: ${({ theme }) => theme.spacing.mobile.md};
  backdrop-filter: blur(4px);

  /* Dark mode styles */
  .dark-mode & {
    background: rgba(0, 0, 0, 0.8) !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 24px;
  }
`;

const ModalContent = styled.div`
  background: #FFFFFF;
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  width: 100%;
  max-width: 600px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
  overflow: hidden;

  /* Dark mode styles */
  .dark-mode & {
    background: #1E293B !important;
  }

  ${({ theme }) => theme.mediaQueries.mobile} {
    max-height: 90vh;
    border-radius: ${({ theme }) => theme.borderRadius.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    max-width: 800px;
    max-height: 85vh;
  }
`;

const ModalHeader = styled.div`
  padding: ${({ theme }) => theme.spacing.mobile.lg};
  border-bottom: 1px solid #F0E6DC;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;

  /* Dark mode styles */
  .dark-mode & {
    border-bottom-color: #374151 !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 24px;
  }
`;

const ModalTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #2C3E50;
  margin: 0;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  /* Dark mode styles */
  .dark-mode & {
    color: #F1F5F9 !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet['2xl']};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 24px;
  }
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  font-size: 24px;
  color: #8B7D7B;
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;

  /* Dark mode styles */
  .dark-mode & {
    color: #94A3B8 !important;
  }

  &:hover {
    background: #F0E6DC;
    color: #2C3E50;

    /* Dark mode styles */
    .dark-mode & {
      background: #374151 !important;
      color: #F1F5F9 !important;
    }
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    width: 44px;
    height: 44px;
  }
`;

const ModalBody = styled.div`
  padding: ${({ theme }) => theme.spacing.mobile.lg};
  overflow-y: auto;
  flex: 1;

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 24px;
  }
`;

const TaskList = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.mobile.md};

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 16px;
  }
`;

const EmptyState = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.mobile.xl};
  color: #8B7D7B;

  /* Dark mode styles */
  .dark-mode & {
    color: #94A3B8 !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 48px;
  }
`;

const EmptyStateIcon = styled.div`
  font-size: 48px;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.lg};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 64px;
    margin-bottom: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 72px;
    margin-bottom: 24px;
  }
`;

const EmptyStateTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #2C3E50;
  margin: 0 0 ${({ theme }) => theme.spacing.mobile.sm} 0;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  /* Dark mode styles */
  .dark-mode & {
    color: #F1F5F9 !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.lg};
    margin: 0 0 ${({ theme }) => theme.spacing.tablet.sm} 0;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 20px;
    margin: 0 0 12px 0;
  }
`;

const EmptyStateMessage = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.base};
  color: #8B7D7B;
  margin: 0;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  line-height: 1.5;

  /* Dark mode styles */
  .dark-mode & {
    color: #94A3B8 !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: ${({ theme }) => theme.typography.fontSize.tablet.base};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 16px;
  }
`;

const LoadingState = styled.div`
  text-align: center;
  padding: ${({ theme }) => theme.spacing.mobile.xl};
  color: #8B7D7B;

  /* Dark mode styles */
  .dark-mode & {
    color: #94A3B8 !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.xl};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 48px;
  }
`;

const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 4px solid #F0E6DC;
  border-top: 4px solid #7FA870;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto ${({ theme }) => theme.spacing.mobile.lg} 0;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    width: 48px;
    height: 48px;
    margin: 0 auto ${({ theme }) => theme.spacing.tablet.lg} 0;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    width: 56px;
    height: 56px;
    margin: 0 auto 24px 0;
  }
`;

const ModalFooter = styled.div`
  padding: ${({ theme }) => theme.spacing.mobile.lg};
  border-top: 1px solid #F0E6DC;
  display: flex;
  justify-content: flex-end;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  flex-shrink: 0;

  /* Dark mode styles */
  .dark-mode & {
    border-top-color: #374151 !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.lg};
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 24px;
    gap: 12px;
  }
`;

const CreateTaskButton = styled(Button)`
  width: 100%;
  margin-top: ${({ theme }) => theme.spacing.mobile.lg};

  ${({ theme }) => theme.mediaQueries.tablet} {
    margin-top: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    margin-top: 24px;
  }
`;

interface TaskSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onTaskSelect: (taskId: string) => void;
  tasks: Task[]; // Pass tasks as props instead of fetching them
  isLoading?: boolean;
  error?: any;
  onCreateTask?: () => void; // Optional callback for creating new task
}

export const TaskSelectionModal: React.FC<TaskSelectionModalProps> = ({
  isOpen,
  onClose,
  onTaskSelect,
  tasks,
  isLoading = false,
  error,
  onCreateTask,
}) => {
  // Filter and sort available tasks
  const availableTasks = useMemo(() => {
    const filtered = tasks
      .filter(task => {
        // Only exclude completed tasks
        if (task.status === 'COMPLETED') {
          return false;
        }

        // Allow all TODO and IN_PROGRESS tasks regardless of pomodoros
        if (task.status === 'TODO' || task.status === 'IN_PROGRESS') {
          return true;
        }

        return false;
      })
      .sort((a, b) => {
        // Sort by priority first
        const priorityOrder = { URGENT: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
        const aPriority = priorityOrder[a.priority];
        const bPriority = priorityOrder[b.priority];

        if (aPriority !== bPriority) {
          return bPriority - aPriority; // Higher priority first
        }

        // Then by progress (least progress first)
        const aProgress = (a.completedPomodoros / (a.estimatedPomodoros || 1)) * 100;
        const bProgress = (b.completedPomodoros / (b.estimatedPomodoros || 1)) * 100;

        return aProgress - bProgress;
      });

    return filtered;
  }, [tasks]);

  const handleTaskSelect = (taskId: string) => {
    onTaskSelect(taskId);
    onClose();
  };

  const handleClose = () => {
    onClose();
  };

  const handleCreateTask = () => {
    onClose();
    if (onCreateTask) {
      onCreateTask();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleClose();
    }
  };

  // Don't render if not open
  if (!isOpen) return null;

  return (
    <ModalOverlay onClick={handleClose} onKeyDown={handleKeyDown}>
      <ModalContent onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="task-selection-title">
        <ModalHeader>
          <ModalTitle id="task-selection-title">Select a Task</ModalTitle>
          <CloseButton onClick={handleClose} aria-label="Close modal">
            ×
          </CloseButton>
        </ModalHeader>

        <ModalBody>
          {isLoading ? (
            <LoadingState>
              <LoadingSpinner />
              <EmptyStateTitle>Loading Tasks...</EmptyStateTitle>
              <EmptyStateMessage>Please wait while we fetch your available tasks.</EmptyStateMessage>
            </LoadingState>
          ) : error ? (
            <EmptyState>
              <EmptyStateIcon>❌</EmptyStateIcon>
              <EmptyStateTitle>Error Loading Tasks</EmptyStateTitle>
              <EmptyStateMessage>
                {typeof error === 'string' ? error : 'Unable to load tasks. Please try again.'}
              </EmptyStateMessage>
            </EmptyState>
          ) : availableTasks.length === 0 ? (
            <EmptyState>
              <EmptyStateIcon>📝</EmptyStateIcon>
              <EmptyStateTitle>No Available Tasks</EmptyStateTitle>
              <EmptyStateMessage>
                You don't have any tasks with remaining pomodoros. Create a new task to get started!
              </EmptyStateMessage>
              {onCreateTask && (
                <CreateTaskButton onClick={handleCreateTask}>
                  ➕ Create New Task
                </CreateTaskButton>
              )}
            </EmptyState>
          ) : (
            <TaskList>
              {availableTasks.map((task) => (
                <div
                  key={task.id}
                  onClick={() => handleTaskSelect(task.id)}
                  style={{ cursor: 'pointer' }}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleTaskSelect(task.id);
                    }
                  }}
                  aria-label={`Select task: ${task.title}`}
                >
                  <TaskCard
                    {...task}
                    onUpdate={undefined} // Disable editing in selection modal
                    onDelete={undefined} // Disable deletion in selection modal
                  />
                </div>
              ))}
            </TaskList>
          )}
        </ModalBody>

        {availableTasks.length > 0 && (
          <ModalFooter>
            <Button variant="secondary" onClick={handleClose}>
              Cancel
            </Button>
          </ModalFooter>
        )}
      </ModalContent>
    </ModalOverlay>
  );
};

export type { TaskSelectionModalProps };