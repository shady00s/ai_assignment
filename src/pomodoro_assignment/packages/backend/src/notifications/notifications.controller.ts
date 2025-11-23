import {
  Controller,
  Get,
  Post,
  Patch,
  Delete,
  Param,
  Body,
  Query,
  UseGuards,
  Request,
  UseInterceptors,
} from '@nestjs/common';
import { NotificationsService } from './notifications.service';
import { CreateNotificationDto } from './dto/create-notification.dto';
import { NotificationResponseDto } from './dto/notification-response.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import {
  ApiStandardResponse,
  ApiStandardArrayResponse,
  ApiDeleteResponse
} from '../common/decorators/api-response.decorator';
import {
  ApiTags,
  ApiOperation,
  ApiParam,
  ApiQuery,
} from '@nestjs/swagger';

@ApiTags('notifications')
@Controller('notifications')
@UseGuards(JwtAuthGuard)
export class NotificationsController {
  constructor(private readonly notificationsService: NotificationsService) {}

  @Post()
  @ApiOperation({ summary: 'Create a new notification' })
  @ApiStandardResponse(NotificationResponseDto, 'Notification created successfully')
  create(@Body() createNotificationDto: CreateNotificationDto, @Request() req) {
    return this.notificationsService.create(createNotificationDto, req.user.id);
  }

  @Get()
  @ApiOperation({ summary: 'Get all notifications for the authenticated user' })
  @ApiQuery({ name: 'unreadOnly', required: false, description: 'Filter by unread notifications only' })
  @ApiQuery({ name: 'type', required: false, description: 'Filter by notification type' })
  @ApiQuery({ name: 'limit', required: false, description: 'Maximum number of notifications to return' })
  @ApiQuery({ name: 'offset', required: false, description: 'Number of notifications to skip' })
  @ApiStandardArrayResponse(NotificationResponseDto, 'Notifications retrieved successfully')
  findAll(
    @Request() req,
    @Query('unreadOnly') unreadOnly?: string,
    @Query('type') type?: string,
    @Query('limit') limit?: string,
    @Query('offset') offset?: string,
  ) {
    return this.notificationsService.findAll(req.user.id, {
      unreadOnly: unreadOnly === 'true',
      type: type as any,
      limit: limit ? parseInt(limit) : undefined,
      offset: offset ? parseInt(offset) : undefined,
    });
  }

  @Get('unread-count')
  @ApiOperation({ summary: 'Get count of unread notifications' })
  getUnreadCount(@Request() req) {
    return this.notificationsService.getUnreadCount(req.user.id);
  }

  @Patch(':id/read')
  @ApiOperation({ summary: 'Mark a notification as read' })
  @ApiParam({ name: 'id', description: 'Notification ID' })
  @ApiStandardResponse(NotificationResponseDto, 'Notification marked as read')
  markAsRead(@Param('id') id: string, @Request() req) {
    return this.notificationsService.markAsRead(id, req.user.id);
  }

  @Patch('read-all')
  @ApiOperation({ summary: 'Mark all notifications as read' })
  markAllAsRead(@Request() req) {
    return this.notificationsService.markAllAsRead(req.user.id);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Delete a notification' })
  @ApiParam({ name: 'id', description: 'Notification ID' })
  @ApiDeleteResponse('Notification deleted successfully')
  remove(@Param('id') id: string, @Request() req) {
    return this.notificationsService.delete(id, req.user.id);
  }
}