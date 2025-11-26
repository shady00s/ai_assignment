import React, { useState } from 'react';
import styled from 'styled-components';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Card } from '../../atoms/Card';
import { Button } from '../../atoms/Button';
import { Input } from '../../atoms/Input';

const EditableTaskCard = styled(Card)<{ $isDragging: boolean; $isEditing: boolean; $priority: string }>`
  cursor: ${({ $isDragging }) => ($isDragging ? 'grabbing' : 'grab')};
  padding: ${({ theme }) => theme.spacing.mobile.md};
  border: 2px solid ${({ $priority, theme }) => {
    switch ($priority) {
      case 'URGENT':
        return theme.colors.error;
      case 'HIGH':
        return theme.colors.warning;
      case 'MEDIUM':
        return theme.colors.accent.main;
      case 'LOW':
        return theme.colors.success;
      default:
        return theme.colors.primary.main;
    }
  }};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  background: white;

  /* Dark mode styles */
  .dark-mode & {
    background: #1E293B !important;
  }
  min-height: 120px;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  transition: all 0.2s ease;
  box-shadow: ${({ $isDragging }) =>
    $isDragging
      ? '0 8px 24px rgba(0, 0, 0, 0.15)'
      : '0 2px 8px rgba(0, 0, 0, 0.08)'
  };
  position: relative;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  overflow: hidden;

  ${({ theme }) => theme.mediaQueries.mobile} {
    padding: 12px;
    gap: 8px;
    min-height: 100px;

    /* Extra small screen adjustments */
    @media (max-width: 380px) {
      padding: 10px;
      gap: 6px;
      min-height: 90px;
    }
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md};
    gap: ${({ theme }) => theme.spacing.tablet.sm};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 16px;
    gap: 8px;
  }

  &:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  }
`;

const TaskHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.xs};
`;

const StatusIcon = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.lg};

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 16px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 16px;
  }
`;

const PriorityBadge = styled.div<{ $color: string }>`
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  background-color: ${({ $color }) => `${$color}20`};
  color: ${({ $color }) => $color};
  text-transform: uppercase;
`;

const TaskTitle = styled.h4`
  margin: 0;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: #2C3E50;
  line-height: 1.3;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  /* Dark mode styles */
  .dark-mode & {
    color: #F1F5F9 !important;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 14px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 14px;
  }
`;

const TaskDescription = styled.p`
  margin: 0;
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: #8B7D7B;
  line-height: 1.4;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};

  /* Dark mode styles */
  .dark-mode & {
    color: #94A3B8 !important;
  }
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 12px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 12px;
  }
`;

const EditableInput = styled(Input)`
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.sm};
  width: 100%;
  box-sizing: border-box;

  ${({ theme }) => theme.mediaQueries.mobile} {
    margin-bottom: 6px;
    font-size: 14px;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 14px;
    margin-bottom: 8px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 14px;
    margin-bottom: 8px;
  }

  /* Extra small screens */
  @media (max-width: 380px) {
    font-size: 13px;
    padding: 8px 10px;
    margin-bottom: 4px;
  }
`;

const EditableTextarea = styled.textarea`
  width: 100%;
  min-height: 60px;
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  border: 1px solid ${({ theme }) => theme.colors.neutral[300]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: ${({ theme }) => theme.colors.neutral[500]};
  background-color: #FFFFFF;

  /* Dark mode styles */
  .dark-mode & {
    background-color: #0F172A !important;
    color: #E2E8F0 !important;
    border-color: #475569 !important;
  }
  line-height: 1.4;
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  resize: vertical;
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
  box-sizing: border-box;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary.main};
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.primary.light}33;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 12px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 12px;
  }
`;

const EditSelect = styled.select`
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  border: 1px solid ${({ theme }) => theme.colors.neutral[300]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  color: #2C3E50;
  background-color: white;

  /* Dark mode styles */
  .dark-mode & {
    background-color: #0F172A !important;
    color: #E2E8F0 !important;
    border-color: #475569 !important;
  }
  margin-bottom: ${({ theme }) => theme.spacing.mobile.sm};
  width: 100%;
  box-sizing: border-box;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary.main};
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.primary.light}33;
  }

  ${({ theme }) => theme.mediaQueries.mobile} {
    margin-bottom: 6px;
    font-size: 14px;
    padding: 8px 10px;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    font-size: 12px;
    margin-bottom: 8px;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    font-size: 12px;
    margin-bottom: 8px;
  }

  /* Extra small screens */
  @media (max-width: 380px) {
    font-size: 13px;
    padding: 8px 10px;
    margin-bottom: 4px;
  }
`;

const EditButtons = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.mobile.sm};
  margin-top: auto;
  flex-wrap: wrap;
  width: 100%;

  ${({ theme }) => theme.mediaQueries.mobile} {
    gap: 6px;
    margin-top: 8px;
  }

  /* Extra small screens */
  @media (max-width: 380px) {
    gap: 4px;
    margin-top: 6px;
    flex-direction: column;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    gap: ${({ theme }) => theme.spacing.tablet.sm};
    flex-direction: row;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    gap: 8px;
    flex-direction: row;
  }
`;

const FormGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  width: 100%;
  box-sizing: border-box;

  /* Mobile: stack in single column for all small screens */
  ${({ theme }) => theme.mediaQueries.mobile} {
    grid-template-columns: 1fr;
    gap: 6px;
  }

  /* Extra small screens */
  @media (max-width: 380px) {
    grid-template-columns: 1fr;
    gap: 4px;
  }
`;

const FullWidthInput = styled.div`
  grid-column: 1 / -1;
  width: 100%;
  box-sizing: border-box;
`;

const FieldLabel = styled.small`
  font-size: 10px;
  color: #8B7D7B;
  display: block;
  margin-bottom: 2px;

  /* Dark mode styles */
  .dark-mode & {
    color: #94A3B8 !important;
  }
`;

const TagMoreCount = styled.span`
  font-size: 10px;
  color: #8B7D7B;

  /* Dark mode styles */
  .dark-mode & {
    color: #94A3B8 !important;
  }
`;

const FieldHint = styled.div`
  font-size: 11px;
  color: ${({ theme }) => theme.colors.neutral[400]};
  margin-bottom: 4px;
  font-style: italic;
  display: flex;
  align-items: center;
  gap: 4px;
  line-height: 1.2;
  word-break: break-word;
  overflow-wrap: break-word;

  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: 9px;
    margin-bottom: 3px;
    gap: 2px;
  }

  /* Extra small screens */
  @media (max-width: 380px) {
    font-size: 8px;
    margin-bottom: 2px;
    gap: 2px;
    text-align: left;
  }
`;

const HintIcon = styled.span`
  font-size: 12px;
  opacity: 0.7;
  flex-shrink: 0;

  ${({ theme }) => theme.mediaQueries.mobile} {
    font-size: 10px;
  }

  /* Extra small screens */
  @media (max-width: 380px) {
    font-size: 9px;
  }
`;

const EditButton = styled(Button)`
  padding: ${({ theme }) => theme.spacing.mobile.xs} ${({ theme }) => theme.spacing.mobile.sm};
  font-size: ${({ theme }) => theme.typography.fontSize.mobile.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  flex: 1;
  min-width: 0;

  ${({ theme }) => theme.mediaQueries.mobile} {
    padding: 6px 8px;
    font-size: 10px;
    flex: 1;
  }

  /* Extra small screens */
  @media (max-width: 380px) {
    padding: 8px 6px;
    font-size: 10px;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-height: 32px;
  }

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: 4px 8px;
    font-size: 10px;
    flex: none;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: 4px 8px;
    font-size: 10px;
    flex: none;
  }
`;

const TagsContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 4px;
`;

const Tag = styled.span`
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  background-color: #F0E6DC;
  color: #8B7D7B;

  /* Dark mode styles */
  .dark-mode & {
    background-color: #374151 !important;
    color: #D1D5DB !important;
  }
`;

const ProgressContainer = styled.div`
  margin-top: auto;
  padding-top: ${({ theme }) => theme.spacing.mobile.sm};
`;

const ProgressHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
  font-size: 11px;
  color: #8B7D7B;

  /* Dark mode styles */
  .dark-mode & {
    color: #94A3B8 !important;
  }
`;

const ProgressBar = styled.div`
  height: 4px;
  background-color: #F0E6DC;
  border-radius: 2px;
  overflow: hidden;

  /* Dark mode styles */
  .dark-mode & {
    background-color: #374151 !important;
  }
`;

const ProgressFill = styled.div<{ $color: string; $percentage: number }>`
  height: 100%;
  background-color: ${({ $color }) => $color};
  border-radius: 2px;
  width: ${({ $percentage }) => $percentage}%;
  transition: width 0.3s ease;
`;

const EditIconButton = styled.button`
  position: absolute;
  top: 8px;
  right: 8px;
  background: none;
  border: none;
  color: #8B7D7B;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  font-size: 12px;

  /* Dark mode styles */
  .dark-mode & {
    color: #94A3B8 !important;
  }

  &:hover {
    background-color: #F0E6DC;
    color: #2C3E50;

    /* Dark mode styles */
    .dark-mode & {
      background-color: #374151 !important;
      color: #F1F5F9 !important;
    }
  }
`;

interface TaskCardProps {
  id: string;
  title: string;
  description?: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';
  status: 'TODO' | 'IN_PROGRESS' | 'DONE';
  estimatedPomodoros?: number;
  completedPomodoros?: number;
  tags?: string[];
  className?: string;
  onUpdate?: (updatedTask: Partial<TaskCardProps>) => void;
  onDelete?: (id: string) => void;
}

export const TaskCard: React.FC<TaskCardProps> = ({
  id,
  title,
  description,
  priority,
  status,
  estimatedPomodoros = 1,
  completedPomodoros = 0,
  tags = [],
  className,
  onUpdate,
  onDelete,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    title,
    description: description || '',
    priority,
    estimatedPomodoros,
    completedPomodoros,
    tags: tags.join(', '),
  });

  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const getPriorityColor = () => {
    switch (priority) {
      case 'URGENT':
        return '#C85A5A'; // Soft Red
      case 'HIGH':
        return '#F4A261'; // Warm Amber
      case 'MEDIUM':
        return '#E9C46A'; // Soft Yellow
      case 'LOW':
        return '#7FA870'; // Sage Green
      default:
        return '#7FA870';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'TODO':
        return '📝';
      case 'IN_PROGRESS':
        return '🔄';
      case 'DONE':
        return '✅';
      default:
        return '📝';
    }
  };

  const getPriorityLabel = () => {
    switch (priority) {
      case 'URGENT':
        return 'Urgent';
      case 'HIGH':
        return 'High';
      case 'MEDIUM':
        return 'Medium';
      case 'LOW':
        return 'Low';
      default:
        return 'Low';
    }
  };

  const progressPercentage = estimatedPomodoros > 0
    ? Math.min((completedPomodoros / estimatedPomodoros) * 100, 100)
    : 0;

  const handleSave = () => {
    const updatedTask = {
      title: editForm.title,
      description: editForm.description || undefined,
      priority: editForm.priority,
      estimatedPomodoros: Number(editForm.estimatedPomodoros) || 1,
      completedPomodoros: Number(editForm.completedPomodoros) || 0,
      tags: editForm.tags ? editForm.tags.split(',').map(tag => tag.trim()).filter(tag => tag) : [],
    };

    if (onUpdate) {
      onUpdate(updatedTask);
    }
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditForm({
      title,
      description: description || '',
      priority,
      estimatedPomodoros,
      completedPomodoros,
      tags: tags.join(', '),
    });
    setIsEditing(false);
  };

  const handleDelete = () => {
    if (onDelete && window.confirm('Are you sure you want to delete this task?')) {
      onDelete(id);
    }
  };

  const handleCardClick = () => {
    if (!isEditing && !isDragging) {
      setIsEditing(true);
    }
  };

  const handleEditButtonClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsEditing(true);
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={className}
      {...attributes}
      {...listeners}
      onClick={handleCardClick}
    >
      <EditableTaskCard
        $isDragging={isDragging}
        $isEditing={isEditing}
        $priority={priority}
        onClick={handleCardClick}
      >
        {!isEditing ? (
          <>
            {/* Edit button */}
            <EditIconButton
              onClick={handleEditButtonClick}
              title="Edit task"
            >
              ✏️
            </EditIconButton>

            {/* Header with status icon and priority */}
            <TaskHeader>
              <StatusIcon>{getStatusIcon()}</StatusIcon>
              <PriorityBadge $color={getPriorityColor()}>
                {getPriorityLabel()}
              </PriorityBadge>
            </TaskHeader>

            {/* Task Title */}
            <TaskTitle>{title}</TaskTitle>

            {/* Task Description (truncated) */}
            {description && (
              <TaskDescription>{description}</TaskDescription>
            )}

            {/* Tags */}
            {tags.length > 0 && (
              <TagsContainer>
                {tags.slice(0, 3).map((tag, index) => (
                  <Tag key={index}>{tag}</Tag>
                ))}
                {tags.length > 3 && (
                  <TagMoreCount>
                    +{tags.length - 3}
                  </TagMoreCount>
                )}
              </TagsContainer>
            )}

            {/* Pomodoro Progress */}
            <ProgressContainer>
              <ProgressHeader>
                <span>Progress</span>
                <span>{completedPomodoros}/{estimatedPomodoros} 🍅</span>
              </ProgressHeader>
              <ProgressBar>
                <ProgressFill $color={getPriorityColor()} $percentage={progressPercentage} />
              </ProgressBar>
            </ProgressContainer>
          </>
        ) : (
          <>
            {/* Edit Form */}
            {/* Task Title with hint */}
            <FieldHint>
              <HintIcon>💡</HintIcon>
              Give your task a clear, action-oriented title
            </FieldHint>
            <EditableInput
              value={editForm.title}
              onChange={(e) => setEditForm({ ...editForm, title: e.target.value })}
              placeholder="Task title"
            />

            {/* Task Description with hint */}
            <FieldHint>
              <HintIcon>📝</HintIcon>
              Add details about what needs to be done (optional)
            </FieldHint>
            <EditableTextarea
              value={editForm.description}
              onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
              placeholder="Task description (optional)"
              onMouseDown={(e) => e.stopPropagation()}
            />

            {/* Priority and Pomodoro inputs with hints */}
            <div>
              <FieldHint>
                <HintIcon>🎯</HintIcon>
                Set task priority and estimate focus sessions
              </FieldHint>
              <FormGrid >
                <div>
                  <FieldLabel>Priority</FieldLabel>
                  <EditSelect
                    value={editForm.priority}
                    onChange={(e) => setEditForm({ ...editForm, priority: e.target.value as TaskCardProps['priority'] })}
                    onMouseDown={(e) => e.stopPropagation()}
                  >
                    <option value="LOW">Low</option>
                    <option value="MEDIUM">Medium</option>
                    <option value="HIGH">High</option>
                    <option value="URGENT">Urgent</option>
                  </EditSelect>
                </div>

                <div>
                  <FieldLabel>Est. Pomodoros</FieldLabel>
                  <EditableInput
                    type="number"
                    value={editForm.estimatedPomodoros.toString()}
                    onChange={(e) => setEditForm({ ...editForm, estimatedPomodoros: Number(e.target.value) || 1 })}
                    placeholder="Est. pomodoros"
                  />
                </div>

                {/* Actual pomodoros input - full width on mobile, second column on desktop */}
                <FullWidthInput>
                  <FieldLabel>Actual Pomodoros</FieldLabel>
                  <EditableInput
                    type="number"
                    value={editForm.completedPomodoros.toString()}
                    onChange={(e) => setEditForm({ ...editForm, completedPomodoros: Number(e.target.value) || 0 })}
                    placeholder="Actual pomodoros"
                  />
                </FullWidthInput>
              </FormGrid>
            </div>

            {/* Tags field with hint */}
            <FieldHint>
              <HintIcon>🏷️</HintIcon>
              Add tags to organize and categorize your tasks (comma separated)
            </FieldHint>
            <EditableInput
              value={editForm.tags}
              onChange={(e) => setEditForm({ ...editForm, tags: e.target.value })}
              placeholder="Tags (comma separated)"
            />

            <EditButtons>
              <EditButton
                variant="primary"
                onClick={handleSave}
              >
                Save
              </EditButton>
              <EditButton
                variant="secondary"
                onClick={handleCancel}
              >
                Cancel
              </EditButton>
              {onDelete && (
                <EditButton
                  variant="ghost"
                  onClick={handleDelete}
                >
                  Delete
                </EditButton>
              )}
            </EditButtons>
          </>
        )}
      </EditableTaskCard>
    </div>
  );
};

export type { TaskCardProps };