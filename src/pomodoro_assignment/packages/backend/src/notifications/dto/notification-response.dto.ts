import { ApiProperty } from '@nestjs/swagger';
import { NotificationType } from './create-notification.dto';

export class NotificationResponseDto {
  @ApiProperty({ description: 'Notification ID' })
  id: string;

  @ApiProperty({ description: 'User ID' })
  userId: string;

  @ApiProperty({
    description: 'Notification type',
    enum: NotificationType
  })
  type: NotificationType;

  @ApiProperty({ description: 'Notification title' })
  title: string;

  @ApiProperty({ description: 'Notification message' })
  message: string;

  @ApiProperty({ description: 'Whether the notification is read' })
  read: boolean;

  @ApiProperty({ description: 'Related entity ID', required: false })
  entityId?: string;

  @ApiProperty({ description: 'Related entity type', required: false })
  entityType?: string;

  @ApiProperty({ description: 'Additional notification data', required: false })
  data?: Record<string, any>;

  @ApiProperty({ description: 'Creation timestamp' })
  createdAt: string;

  @ApiProperty({ description: 'Read timestamp', required: false })
  readAt?: string;

  @ApiProperty({ description: 'Scheduled delivery time', required: false })
  scheduledFor?: string;
}