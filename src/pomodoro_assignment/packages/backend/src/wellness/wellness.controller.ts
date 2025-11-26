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
  WellnessRecommendationDto,
  // New DTOs for additional endpoints
  IncrementHydrationDto,
  SetHydrationGoalDto,
  LogMovementDto,
  LogStepsDto,
  SetMovementGoalDto,
  UpdateMoodDto,
  LogSleepDto,
  LogMeditationDto,
  CompleteMeditationDto,
  LogBreathingDto,
  LogPostureCheckDto,
  LogEyeRestDto,
  QuickWaterDto,
  QuickMovementDto,
  QuickMoodDto,
  QuickMeditationDto,
  AcknowledgeRecommendationDto,
  WellnessScoreResponseDto,
  WellnessAchievementDto
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

  // ========================================
  // HYDRATION ENDPOINTS
  // ========================================

  @Post('hydration/increment')
  @ApiOperation({ summary: 'Increment water intake for today' })
  @ApiResponse({
    status: 200,
    description: 'Hydration incremented successfully',
    type: WellnessEntryResponseDto
  })
  async incrementHydration(
    @Request() req,
    @Body(ValidationPipe) incrementData: IncrementHydrationDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.incrementHydration(userId, incrementData);
  }

  @Post('hydration/goal')
  @ApiOperation({ summary: 'Set daily hydration goal' })
  @ApiResponse({
    status: 200,
    description: 'Hydration goal set successfully',
    type: WellnessEntryResponseDto
  })
  async setHydrationGoal(
    @Request() req,
    @Body(ValidationPipe) goalData: SetHydrationGoalDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.setHydrationGoal(userId, goalData);
  }

  // ========================================
  // MOVEMENT ENDPOINTS
  // ========================================

  @Post('movement/log')
  @ApiOperation({ summary: 'Log a movement break' })
  @ApiResponse({
    status: 200,
    description: 'Movement break logged successfully',
    type: WellnessEntryResponseDto
  })
  async logMovementBreak(
    @Request() req,
    @Body(ValidationPipe) movementData: LogMovementDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.logMovementBreak(userId, movementData);
  }

  @Post('movement/steps')
  @ApiOperation({ summary: 'Log step count for a specific date' })
  @ApiResponse({
    status: 200,
    description: 'Steps logged successfully',
    type: WellnessEntryResponseDto
  })
  async logSteps(
    @Request() req,
    @Body(ValidationPipe) stepsData: LogStepsDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.logSteps(userId, stepsData);
  }

  @Post('movement/goal')
  @ApiOperation({ summary: 'Set daily movement goals' })
  @ApiResponse({
    status: 200,
    description: 'Movement goals set successfully'
  })
  async setMovementGoals(
    @Request() req,
    @Body(ValidationPipe) goalsData: SetMovementGoalDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.setMovementGoals(userId, goalsData);
  }

  // ========================================
  // MOOD AND MENTAL WELLNESS ENDPOINTS
  // ========================================

  @Post('mood/update')
  @ApiOperation({ summary: 'Update mood, stress, and energy levels' })
  @ApiResponse({
    status: 200,
    description: 'Mood updated successfully',
    type: WellnessEntryResponseDto
  })
  async updateMood(
    @Request() req,
    @Body(ValidationPipe) moodData: UpdateMoodDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.updateMood(userId, moodData);
  }

  @Post('mood/sleep')
  @ApiOperation({ summary: 'Log sleep hours and quality' })
  @ApiResponse({
    status: 200,
    description: 'Sleep data logged successfully',
    type: WellnessEntryResponseDto
  })
  async logSleep(
    @Request() req,
    @Body(ValidationPipe) sleepData: LogSleepDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.logSleep(userId, sleepData);
  }

  // ========================================
  // MEDITATION ENDPOINTS
  // ========================================

  @Post('meditation/log')
  @ApiOperation({ summary: 'Log meditation session' })
  @ApiResponse({
    status: 200,
    description: 'Meditation session logged successfully',
    type: WellnessEntryResponseDto
  })
  async logMeditation(
    @Request() req,
    @Body(ValidationPipe) meditationData: LogMeditationDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.logMeditation(userId, meditationData);
  }

  @Post('meditation/complete')
  @ApiOperation({ summary: 'Complete a meditation session' })
  @ApiResponse({
    status: 200,
    description: 'Meditation session completed successfully'
  })
  async completeMeditationSession(
    @Request() req,
    @Body(ValidationPipe) sessionData: CompleteMeditationDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.completeMeditationSession(userId, sessionData);
  }

  @Post('meditation/breathing')
  @ApiOperation({ summary: 'Log breathing exercise' })
  @ApiResponse({
    status: 200,
    description: 'Breathing exercise logged successfully',
    type: WellnessEntryResponseDto
  })
  async logBreathingExercise(
    @Request() req,
    @Body(ValidationPipe) breathingData: LogBreathingDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.logBreathingExercise(userId, breathingData);
  }

  // ========================================
  // POSTURE AND EYE REST ENDPOINTS
  // ========================================

  @Post('posture/check')
  @ApiOperation({ summary: 'Log posture check completion' })
  @ApiResponse({
    status: 200,
    description: 'Posture check logged successfully',
    type: WellnessEntryResponseDto
  })
  async logPostureCheck(
    @Request() req,
    @Body(ValidationPipe) postureData: LogPostureCheckDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.logPostureCheck(userId, postureData);
  }

  @Post('eye-rest/break')
  @ApiOperation({ summary: 'Log eye rest break' })
  @ApiResponse({
    status: 200,
    description: 'Eye rest break logged successfully',
    type: WellnessEntryResponseDto
  })
  async logEyeRestBreak(
    @Request() req,
    @Body(ValidationPipe) eyeRestData: LogEyeRestDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.logEyeRestBreak(userId, eyeRestData);
  }

  // ========================================
  // QUICK ACTION ENDPOINTS
  // ========================================

  @Post('quick/water')
  @ApiOperation({ summary: 'Quick water logging' })
  @ApiResponse({
    status: 200,
    description: 'Water logged successfully',
    type: WellnessEntryResponseDto
  })
  async quickLogWater(
    @Request() req,
    @Body(ValidationPipe) waterData: QuickWaterDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.quickLogWater(userId, waterData);
  }

  @Post('quick/movement')
  @ApiOperation({ summary: 'Quick movement logging' })
  @ApiResponse({
    status: 200,
    description: 'Movement logged successfully',
    type: WellnessEntryResponseDto
  })
  async quickLogMovement(
    @Request() req,
    @Body(ValidationPipe) movementData: QuickMovementDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.quickLogMovement(userId, movementData);
  }

  @Post('quick/mood')
  @ApiOperation({ summary: 'Quick mood logging' })
  @ApiResponse({
    status: 200,
    description: 'Mood logged successfully',
    type: WellnessEntryResponseDto
  })
  async quickLogMood(
    @Request() req,
    @Body(ValidationPipe) moodData: QuickMoodDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.quickLogMood(userId, moodData);
  }

  @Post('quick/meditation')
  @ApiOperation({ summary: 'Quick meditation logging' })
  @ApiResponse({
    status: 200,
    description: 'Meditation logged successfully',
    type: WellnessEntryResponseDto
  })
  async quickLogMeditation(
    @Request() req,
    @Body(ValidationPipe) meditationData: QuickMeditationDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.quickLogMeditation(userId, meditationData);
  }

  // ========================================
  // UTILITY ENDPOINTS
  // ========================================

  @Get('score')
  @ApiOperation({ summary: 'Get current wellness score with breakdown' })
  @ApiResponse({
    status: 200,
    description: 'Wellness score retrieved successfully',
    type: WellnessScoreResponseDto
  })
  async getWellnessScore(@Request() req) {
    const userId = req.user.id;
    return await this.wellnessService.getWellnessScore(userId);
  }

  @Get('achievements')
  @ApiOperation({ summary: 'Get wellness-related achievements' })
  @ApiResponse({
    status: 200,
    description: 'Wellness achievements retrieved successfully',
    type: [WellnessAchievementDto]
  })
  async getWellnessAchievements(@Request() req) {
    const userId = req.user.id;
    return await this.wellnessService.getWellnessAchievements(userId);
  }

  @Post('recommendations/:recommendationId/acknowledge')
  @ApiOperation({ summary: 'Acknowledge or unacknowledge a recommendation' })
  @ApiParam({
    name: 'recommendationId',
    description: 'ID of the recommendation to acknowledge',
    example: 'wellness_rec_001'
  })
  @ApiResponse({
    status: 200,
    description: 'Recommendation acknowledged successfully'
  })
  async acknowledgeRecommendation(
    @Request() req,
    @Param('recommendationId') recommendationId: string,
    @Body(ValidationPipe) acknowledgeData: AcknowledgeRecommendationDto,
  ) {
    const userId = req.user.id;
    return await this.wellnessService.acknowledgeRecommendation(userId, recommendationId, acknowledgeData);
  }
}