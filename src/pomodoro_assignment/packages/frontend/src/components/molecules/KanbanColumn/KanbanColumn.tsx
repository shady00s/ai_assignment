import React from 'react';
import styled from 'styled-components';
import { useDroppable } from '@dnd-kit/core';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { TaskCard } from '../TaskCard';

const ColumnContainer = styled.div<{ $isOver: boolean; $color: string }>`
  display: flex;
  flex-direction: column;
  flex: 0 0 auto;
  width: 85vw;
  height: 65vh;
  max-width: 320px;
  min-height: 400px;
  background-color: ${props => props.$isOver ? `${props.$color}15` : '#F8F9FA'};
  border-radius: 16px;
  border: ${props => props.$isOver ? `2px solid ${props.$color}` : '1px solid rgba(0,0,0,0.08)'};
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  scroll-snap-align: start;
  position: relative;

  ${props => props.theme.mediaQueries.mobile} {
    width: 80vw;
    max-width: 300px;
    height: 60vh;
    min-height: 350px;
    transform: translateX(0);
    will-change: transform;
  }

  ${props => props.theme.mediaQueries.tablet} {
    width: 45vw;
    max-width: 320px;
    height: 550px;
    min-height: 450px;
  }

  ${props => props.theme.mediaQueries.desktop} {
    width: 320px;
    max-width: 350px;
    height: auto;
    min-height: 500px;
    flex: 1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  }

  /* Mobile touch enhancements */
  ${props => props.theme.mediaQueries.mobile} {
    &:active {
      transform: scale(0.98);
    }
  }
`;

const ColumnHeader = styled.div<{ $color: string }>`
  padding: ${props => props.theme.spacing.mobile.sm} ${props => props.theme.spacing.mobile.md};
  border-bottom: 1px solid ${props => `${props.$color}20`};
  border-radius: 16px 16px 0 0;
  background-color: white;
  flex-shrink: 0;

  ${props => props.theme.mediaQueries.tablet} {
    padding: 16px 20px;
  }

  ${props => props.theme.mediaQueries.desktop} {
    padding: 16px 20px;
  }
`;

const HeaderContent = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const TitleSection = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.mobile.sm};

  ${props => props.theme.mediaQueries.tablet} {
    gap: 8px;
  }

  ${props => props.theme.mediaQueries.desktop} {
    gap: 8px;
  }
`;

const ColumnIcon = styled.span`
  font-size: 20px;
  line-height: 1;

  ${props => props.theme.mediaQueries.mobile} {
    font-size: 18px;
  }

  ${props => props.theme.mediaQueries.tablet} {
    font-size: 20px;
  }
`;

const ColumnTitle = styled.h3`
  margin: 0;
  font-size: ${props => props.theme.typography.fontSize.mobile.base};
  font-weight: ${props => props.theme.typography.fontWeight.semibold};
  color: #2C3E50;
  font-family: ${props => props.theme.typography.fontFamily.primary};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  ${props => props.theme.mediaQueries.tablet} {
    font-size: 16px;
  }

  ${props => props.theme.mediaQueries.desktop} {
    font-size: 16px;
  }
`;

const TaskCount = styled.div<{ $color: string }>`
  background-color: ${props => props.$color};
  color: white;
  padding: 6px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: ${props => props.theme.typography.fontWeight.semibold};
  min-width: 28px;
  text-align: center;
  flex-shrink: 0;
`;

const TasksContainer = styled.div`
  flex: 1;
  padding: ${props => props.theme.spacing.mobile.sm};
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.mobile.sm};
  min-height: 0;
  position: relative;

  /* Hide scrollbar for mobile */
  -ms-overflow-style: none;
  scrollbar-width: none;
  &::-webkit-scrollbar {
    display: none;
  }

  ${props => props.theme.mediaQueries.mobile} {
    padding: 12px;
    gap: 10px;

    /* Mobile touch enhancements */
    -webkit-tap-highlight-color: transparent;
    touch-action: pan-y;

    /* Improved scroll performance on mobile */
    transform: translateZ(0);
    -webkit-overflow-scrolling: touch;
  }

  ${props => props.theme.mediaQueries.tablet} {
    padding: 16px;
    gap: 12px;
  }

  ${props => props.theme.mediaQueries.desktop} {
    padding: 16px;
    gap: 12px;

    /* Show scrollbar for desktop */
    &::-webkit-scrollbar {
      display: block;
      width: 6px;
    }

    &::-webkit-scrollbar-track {
      background: #f1f1f1;
      border-radius: 3px;
    }

    &::-webkit-scrollbar-thumb {
      background: #c1c1c1;
      border-radius: 3px;
    }

    &::-webkit-scrollbar-thumb:hover {
      background: #a8a8a8;
    }
  }
`;

const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px 16px;
  text-align: center;
  color: #8B7D7B;
  font-size: ${props => props.theme.typography.fontSize.mobile.sm};
  font-family: ${props => props.theme.typography.fontFamily.primary};
  border: 2px dashed #D4C4B0;
  border-radius: ${props => props.theme.borderRadius.md};
  min-height: 150px;
  margin: auto;
  transition: all 0.3s ease;
  position: relative;

  ${props => props.theme.mediaQueries.mobile} {
    min-height: 120px;
    padding: 24px 12px;

    /* Mobile hover/touch effect */
    &:active {
      transform: scale(0.98);
      background-color: rgba(212, 197, 185, 0.1);
    }
  }

  ${props => props.theme.mediaQueries.tablet} {
    font-size: 14px;
    min-height: 120px;
    padding: 28px 16px;
  }

  ${props => props.theme.mediaQueries.desktop} {
    font-size: 14px;
    padding: 32px 16px;
  }
`;

const EmptyStateIcon = styled.div`
  font-size: 28px;
  margin-bottom: 8px;
  opacity: 0.5;

  ${props => props.theme.mediaQueries.mobile} {
    font-size: 24px;
  }
`;

const EmptyStateTitle = styled.div`
  font-weight: ${props => props.theme.typography.fontWeight.semibold};
  margin-bottom: 4px;
`;

const EmptyStateDescription = styled.div`
  font-size: 12px;
  opacity: 0.7;
  line-height: 1.4;
`;

interface Task {
  id: string;
  title: string;
  description?: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';
  status: 'TODO' | 'IN_PROGRESS' | 'DONE';
  estimatedPomodoros?: number;
  actualPomodoros?: number;
  tags?: string[];
}

interface KanbanColumnProps {
  id: string;
  title: string;
  tasks: Task[];
  onUpdateTask?: (taskId: string, updates: Partial<Task>) => void;
  onDeleteTask?: (taskId: string) => void;
  className?: string;
}

export const KanbanColumn: React.FC<KanbanColumnProps> = ({
  id,
  title,
  tasks,
  onUpdateTask,
  onDeleteTask,
  className,
}) => {
  const { setNodeRef, isOver } = useDroppable({
    id,
  });

  const getColumnColor = () => {
    switch (id) {
      case 'TODO':
        return '#D4C5B9'; // Zen Stone
      case 'IN_PROGRESS':
        return '#F4A261'; // Warm Amber
      case 'DONE':
        return '#7FA870'; // Sage Green
      default:
        return '#D4C5B9';
    }
  };

  const getColumnIcon = () => {
    switch (id) {
      case 'TODO':
        return '📋';
      case 'IN_PROGRESS':
        return '🔄';
      case 'DONE':
        return '✅';
      default:
        return '📋';
    }
  };

  return (
    <ColumnContainer
      className={className}
      $isOver={isOver}
      $color={getColumnColor()}
      data-column-id={id}
    >
      {/* Column Header */}
      <ColumnHeader $color={getColumnColor()}>
        <HeaderContent>
          <TitleSection>
            <ColumnIcon>{getColumnIcon()}</ColumnIcon>
            <ColumnTitle>{title}</ColumnTitle>
          </TitleSection>
          <TaskCount $color={getColumnColor()}>{tasks.length}</TaskCount>
        </HeaderContent>
      </ColumnHeader>

      {/* Tasks Container */}
      <TasksContainer ref={setNodeRef}>
        <SortableContext items={tasks.map(task => task.id)} strategy={verticalListSortingStrategy}>
          {tasks.length === 0 ? (
            <EmptyState>
              <EmptyStateIcon>{getColumnIcon()}</EmptyStateIcon>
              <EmptyStateTitle>No tasks yet</EmptyStateTitle>
              <EmptyStateDescription>Drag tasks here to get started</EmptyStateDescription>
            </EmptyState>
          ) : (
            tasks.map((task) => (
              <TaskCard
                key={task.id}
                {...task}
                onUpdate={(updates) => onUpdateTask?.(task.id, updates)}
                onDelete={() => onDeleteTask?.(task.id)}
              />
            ))
          )}
        </SortableContext>
      </TasksContainer>
    </ColumnContainer>
  );
};

export type { KanbanColumnProps };