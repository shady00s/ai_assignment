import {
  Controller,
  Get,
  Post,
  Put,
  Delete,
  Body,
  Query,
  Param,
  UseGuards,
  Request,
  ValidationPipe,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiParam, ApiQuery, ApiBearerAuth } from '@nestjs/swagger';
import { WellnessService } from './wellness.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import {
  CreateWellnessEntryDto,
  UpdateWellnessEntryDto,
  CreateWellnessReminderDto,
  UpdateWellnessReminderDto,
  CreateWellnessGoalDto,
  UpdateWellnessGoalDto,
  WellnessHistoryQueryDto,
  WellnessAnalyticsQueryDto,
  WellnessEntryResponseDto,
  WellnessReminderResponseDto,
  WellnessGoalResponseDto,
  WellnessAnalyticsDto,
  WellnessTrendsDto,
  WellnessRecommendationDto
} from './dto';

@ApiTags('wellness')
@Controller('wellness')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class WellnessController {
  constructor(private readonly wellnessService: WellnessService) {}

  // ========================================
  // WELLNESS ENTRY ENDPOINTS
  // ========================================

  @Get('today')
  @ApiOperation({ summary: "Get today's wellness entry for the current user" })
  @ApiResponse({
    status: 200,
    description: "Today's wellness entry retrieved successfully",
    schema: {
      oneOf: [
        { type: 'null', description: 'No entry found for today' },
        { $ref: '#/components/schemas/WellnessEntryResponseDto' }
      ]
    }
  })
  async getTodayWellnessEntry(@Request() req) {
    const userId = req.user.id;
    return await this.wellnessService.getTodayWellnessEntry(userId);
  }

  @Post('entry')
  @ApiOperation({ summary: 'Create or update today wellness entry' })
  @ApiResponse({
    status: 201,
    description: 'Wellness entry created/updated successfully',
    type: WellnessEntryResponseDto
  })
  async createOrUpdateWellnessEntry(
    @Request() req,
    @Body(ValidationPipe) createWellnessEntryDto: CreateWellnessEntryDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.createOrUpdateWellnessEntry(userId, createWellnessEntryDto);
  }

  @Put('entry/:date')
  @ApiOperation({ summary: 'Update wellness entry for a specific date' })
  @ApiParam({
    name: 'date',
    description: 'Date in YYYY-MM-DD format',
    example: '2024-01-15'
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness entry updated successfully',
    type: WellnessEntryResponseDto
  })
  async updateWellnessEntryByDate(
    @Request() req,
    @Param('date') date: string,
    @Body(ValidationPipe) updateWellnessEntryDto: UpdateWellnessEntryDto,
  ) {
    const userId = req.user.id;
    const parsedDate = new Date(date);
    if (isNaN(parsedDate.getTime())) {
      throw new Error('Invalid date format. Please use YYYY-MM-DD format.');
    }
    return await this.wellnessService.updateWellnessEntryByDate(userId, parsedDate, updateWellnessEntryDto);
  }

  @Delete('entry/:date')
  @ApiOperation({ summary: 'Delete wellness entry for a specific date' })
  @ApiParam({
    name: 'date',
    description: 'Date in YYYY-MM-DD format',
    example: '2024-01-15'
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness entry deleted successfully'
  })
  async deleteWellnessEntryByDate(
    @Request() req,
    @Param('date') date: string,
  ) {
    const userId = req.user.id;
    const parsedDate = new Date(date);
    if (isNaN(parsedDate.getTime())) {
      throw new Error('Invalid date format. Please use YYYY-MM-DD format.');
    }
    return await this.wellnessService.deleteWellnessEntryByDate(userId, parsedDate);
  }

  @Get('history')
  @ApiOperation({ summary: 'Get wellness history for the current user' })
  @ApiQuery({
    name: 'startDate',
    description: 'Start date in YYYY-MM-DD format',
    required: false,
    example: '2024-01-01'
  })
  @ApiQuery({
    name: 'endDate',
    description: 'End date in YYYY-MM-DD format',
    required: false,
    example: '2024-01-31'
  })
  @ApiQuery({
    name: 'days',
    description: 'Number of days to look back from today',
    required: false,
    example: 30
  })
  @ApiQuery({
    name: 'page',
    description: 'Page number for pagination',
    required: false,
    example: 1
  })
  @ApiQuery({
    name: 'limit',
    description: 'Number of items per page',
    required: false,
    example: 10
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness history retrieved successfully'
  })
  async getWellnessHistory(
    @Request() req,
    @Query() query: WellnessHistoryQueryDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.getWellnessHistory(userId, query);
  }

  // ========================================
  // WELLNESS REMINDER ENDPOINTS
  // ========================================

  @Get('reminders')
  @ApiOperation({ summary: 'Get all wellness reminders for the current user' })
  @ApiResponse({
    status: 200,
    description: 'Wellness reminders retrieved successfully',
    type: [WellnessReminderResponseDto]
  })
  async getWellnessReminders(@Request() req) {
    const userId = req.user.id;
    return await this.wellnessService.getWellnessReminders(userId);
  }

  @Post('reminders')
  @ApiOperation({ summary: 'Create a new wellness reminder' })
  @ApiResponse({
    status: 201,
    description: 'Wellness reminder created successfully',
    type: WellnessReminderResponseDto
  })
  async createWellnessReminder(
    @Request() req,
    @Body(ValidationPipe) createWellnessReminderDto: CreateWellnessReminderDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.createWellnessReminder(userId, createWellnessReminderDto);
  }

  @Put('reminders/:reminderId')
  @ApiOperation({ summary: 'Update a wellness reminder' })
  @ApiParam({
    name: 'reminderId',
    description: 'ID of the reminder to update',
    example: 'clym8d1230000sbdp1234567'
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness reminder updated successfully',
    type: WellnessReminderResponseDto
  })
  async updateWellnessReminder(
    @Request() req,
    @Param('reminderId') reminderId: string,
    @Body(ValidationPipe) updateWellnessReminderDto: UpdateWellnessReminderDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.updateWellnessReminder(userId, reminderId, updateWellnessReminderDto);
  }

  @Delete('reminders/:reminderId')
  @ApiOperation({ summary: 'Delete a wellness reminder' })
  @ApiParam({
    name: 'reminderId',
    description: 'ID of the reminder to delete',
    example: 'clym8d1230000sbdp1234567'
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness reminder deleted successfully'
  })
  async deleteWellnessReminder(
    @Request() req,
    @Param('reminderId') reminderId: string,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.deleteWellnessReminder(userId, reminderId);
  }

  // ========================================
  // WELLNESS GOAL ENDPOINTS
  // ========================================

  @Get('goals')
  @ApiOperation({ summary: 'Get all wellness goals for the current user' })
  @ApiResponse({
    status: 200,
    description: 'Wellness goals retrieved successfully',
    type: [WellnessGoalResponseDto]
  })
  async getWellnessGoals(@Request() req) {
    const userId = req.user.id;
    return await this.wellnessService.getWellnessGoals(userId);
  }

  @Post('goals')
  @ApiOperation({ summary: 'Create a new wellness goal' })
  @ApiResponse({
    status: 201,
    description: 'Wellness goal created successfully',
    type: WellnessGoalResponseDto
  })
  async createWellnessGoal(
    @Request() req,
    @Body(ValidationPipe) createWellnessGoalDto: CreateWellnessGoalDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.createWellnessGoal(userId, createWellnessGoalDto);
  }

  @Put('goals/:goalId')
  @ApiOperation({ summary: 'Update a wellness goal' })
  @ApiParam({
    name: 'goalId',
    description: 'ID of the goal to update',
    example: 'clym8d1230000sbdp1234567'
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness goal updated successfully',
    type: WellnessGoalResponseDto
  })
  async updateWellnessGoal(
    @Request() req,
    @Param('goalId') goalId: string,
    @Body(ValidationPipe) updateWellnessGoalDto: UpdateWellnessGoalDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.updateWellnessGoal(userId, goalId, updateWellnessGoalDto);
  }

  @Delete('goals/:goalId')
  @ApiOperation({ summary: 'Delete a wellness goal' })
  @ApiParam({
    name: 'goalId',
    description: 'ID of the goal to delete',
    example: 'clym8d1230000sbdp1234567'
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness goal deleted successfully'
  })
  async deleteWellnessGoal(
    @Request() req,
    @Param('goalId') goalId: string,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.deleteWellnessGoal(userId, goalId);
  }

  // ========================================
  // WELLNESS ANALYTICS ENDPOINTS
  // ========================================

  @Get('analytics/summary')
  @ApiOperation({ summary: 'Get comprehensive wellness analytics' })
  @ApiQuery({
    name: 'days',
    description: 'Analysis period in days',
    required: false,
    example: 30
  })
  @ApiQuery({
    name: 'startDate',
    description: 'Start date in YYYY-MM-DD format',
    required: false,
    example: '2024-01-01'
  })
  @ApiQuery({
    name: 'endDate',
    description: 'End date in YYYY-MM-DD format',
    required: false,
    example: '2024-01-31'
  })
  @ApiQuery({
    name: 'category',
    description: 'Category to focus analysis on',
    required: false,
    example: 'ALL'
  })
  @ApiQuery({
    name: 'includeRecommendations',
    description: 'Include recommendations in response',
    required: false,
    example: true
  })
  @ApiQuery({
    name: 'includeTrends',
    description: 'Include trends data in response',
    required: false,
    example: false
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness analytics retrieved successfully',
    type: WellnessAnalyticsDto
  })
  async getWellnessAnalytics(
    @Request() req,
    @Query() query: WellnessAnalyticsQueryDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.getWellnessAnalytics(userId, query);
  }

  @Get('analytics/trends')
  @ApiOperation({ summary: 'Get wellness trends over time' })
  @ApiQuery({
    name: 'days',
    description: 'Number of days to analyze',
    required: false,
    example: 30
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness trends retrieved successfully',
    type: [WellnessTrendsDto]
  })
  async getWellnessTrends(
    @Request() req,
    @Query('days') days: string = '30',
  ) {
    const userId = req.user.id;
    const analytics = await this.wellnessService.getWellnessAnalytics(userId, {
      days: parseInt(days),
      includeTrends: true,
      includeRecommendations: false
    });
    return analytics.trends || [];
  }

  @Get('analytics/recommendations')
  @ApiOperation({ summary: 'Get personalized wellness recommendations' })
  @ApiQuery({
    name: 'days',
    description: 'Analysis period in days',
    required: false,
    example: 30
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness recommendations retrieved successfully',
    type: [WellnessRecommendationDto]
  })
  async getWellnessRecommendations(
    @Request() req,
    @Query('days') days: string = '30',
  ) {
    const userId = req.user.id;
    const analytics = await this.wellnessService.getWellnessAnalytics(userId, {
      days: parseInt(days),
      includeRecommendations: true,
      includeTrends: false
    });
    return analytics.recommendations || [];
  }

  @Get('summary')
  @ApiOperation({ summary: 'Get wellness summary for dashboard' })
  @ApiResponse({
    status: 200,
    description: 'Wellness summary retrieved successfully'
  })
  async getWellnessSummary(@Request() req) {
    const userId = req.user.id;
    return await this.wellnessService.getWellnessSummary(userId);
  }
}