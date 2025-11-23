import {
  Controller,
  Get,
  Query,
  Param,
  UseGuards,
  Request,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiParam, ApiQuery, ApiBearerAuth } from '@nestjs/swagger';
import { AnalyticsService } from './analytics.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('analytics')
@Controller('analytics')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class AnalyticsController {
  constructor(private readonly analyticsService: AnalyticsService) {}

  @Get('focus')
  @ApiOperation({ summary: 'Get focus analytics for the current user' })
  @ApiQuery({
    name: 'startDate',
    required: false,
    description: 'Start date for analytics (ISO string)',
    example: '2024-01-01T00:00:00.000Z',
  })
  @ApiQuery({
    name: 'endDate',
    required: false,
    description: 'End date for analytics (ISO string)',
    example: '2024-01-31T23:59:59.999Z',
  })
  @ApiResponse({
    status: 200,
    description: 'Focus analytics retrieved successfully',
    schema: {
      type: 'object',
      properties: {
        dailyFocusTime: { type: 'number', description: 'Minutes of focus time today' },
        weeklyFocusTime: { type: 'number', description: 'Minutes of focus time this week' },
        monthlyFocusTime: { type: 'number', description: 'Minutes of focus time this month' },
        averageSessionLength: { type: 'number', description: 'Average session length in minutes' },
        peakFocusHours: { type: 'array', items: { type: 'number' }, description: 'Hours with most focus time' },
        focusTrend: { type: 'string', enum: ['IMPROVING', 'DECLINING', 'STABLE'] },
        completionRate: { type: 'number', description: 'Session completion rate percentage' },
      },
    },
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getFocusAnalytics(
    @Request() req,
    @Query('startDate') startDate?: string,
    @Query('endDate') endDate?: string,
  ) {
    const start = startDate ? new Date(startDate) : undefined;
    const end = endDate ? new Date(endDate) : undefined;

    return this.analyticsService.getFocusAnalytics(req.user.id, start, end);
  }

  @Get('wellness')
  @ApiOperation({ summary: 'Get wellness analytics for the current user' })
  @ApiQuery({
    name: 'startDate',
    required: false,
    description: 'Start date for analytics (ISO string)',
    example: '2024-01-01T00:00:00.000Z',
  })
  @ApiQuery({
    name: 'endDate',
    required: false,
    description: 'End date for analytics (ISO string)',
    example: '2024-01-31T23:59:59.999Z',
  })
  @ApiResponse({
    status: 200,
    description: 'Wellness analytics retrieved successfully',
    schema: {
      type: 'object',
      properties: {
        mindfulnessMinutes: { type: 'number', description: 'Minutes of mindfulness/breaks' },
        hydrationGoal: { type: 'number', description: 'Daily hydration goal in glasses' },
        hydrationCurrent: { type: 'number', description: 'Current hydration intake' },
        movementGoal: { type: 'number', description: 'Daily movement/break goal' },
        movementCurrent: { type: 'number', description: 'Current movement/break count' },
        moodRating: { type: 'number', minimum: 1, maximum: 5, description: 'Current mood rating' },
        stressLevel: { type: 'number', minimum: 1, maximum: 5, description: 'Current stress level' },
        energyLevel: { type: 'number', minimum: 1, maximum: 5, description: 'Current energy level' },
      },
    },
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getWellnessAnalytics(
    @Request() req,
    @Query('startDate') startDate?: string,
    @Query('endDate') endDate?: string,
  ) {
    const start = startDate ? new Date(startDate) : undefined;
    const end = endDate ? new Date(endDate) : undefined;

    return this.analyticsService.getWellnessAnalytics(req.user.id, start, end);
  }

  @Get('teams/:teamId')
  @ApiOperation({ summary: 'Get team analytics' })
  @ApiParam({
    name: 'teamId',
    description: 'Team ID',
    example: 'team-123',
  })
  @ApiQuery({
    name: 'startDate',
    required: false,
    description: 'Start date for analytics (ISO string)',
    example: '2024-01-01T00:00:00.000Z',
  })
  @ApiQuery({
    name: 'endDate',
    required: false,
    description: 'End date for analytics (ISO string)',
    example: '2024-01-31T23:59:59.999Z',
  })
  @ApiResponse({
    status: 200,
    description: 'Team analytics retrieved successfully',
    schema: {
      type: 'object',
      properties: {
        teamId: { type: 'string', description: 'Team ID' },
        teamName: { type: 'string', description: 'Team name' },
        memberCount: { type: 'number', description: 'Number of team members' },
        totalFocusTime: { type: 'number', description: 'Total focus time for team (minutes)' },
        averageFocusTime: { type: 'number', description: 'Average focus time per member (minutes)' },
        tasksCompleted: { type: 'number', description: 'Total tasks completed by team' },
        averageCompletionRate: { type: 'number', description: 'Average completion rate percentage' },
        topPerformers: { type: 'array', description: 'Top performing members' },
        focusTrend: { type: 'string', enum: ['IMPROVING', 'DECLINING', 'STABLE'] },
        wellnessScore: { type: 'number', description: 'Team average wellness score' },
        collaborationScore: { type: 'number', description: 'Team collaboration score' },
        period: { type: 'object', description: 'Analytics period' },
      },
    },
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 403,
    description: 'Access denied - not a team member',
  })
  @ApiResponse({
    status: 404,
    description: 'Team not found',
  })
  async getTeamAnalytics(
    @Param('teamId') teamId: string,
    @Request() req,
    @Query('startDate') startDate?: string,
    @Query('endDate') endDate?: string,
  ) {
    const start = startDate ? new Date(startDate) : undefined;
    const end = endDate ? new Date(endDate) : undefined;

    return this.analyticsService.getTeamAnalytics(teamId, start, end, req.user.id);
  }
}