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
import {
  FocusAnalyticsDto,
  WellnessAnalyticsDto,
  TeamAnalyticsDto,
  AnalyticsQueryDto
} from './dto';

@ApiTags('analytics')
@Controller('analytics')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class AnalyticsController {
  constructor(private readonly analyticsService: AnalyticsService) {}

  @Get('focus')
  @ApiOperation({ summary: 'Get focus analytics for the current user' })
  @ApiResponse({
    status: 200,
    description: 'Focus analytics retrieved successfully',
    type: FocusAnalyticsDto
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getFocusAnalytics(
    @Request() req,
    @Query() query: AnalyticsQueryDto,
  ) {
    const { startDate, endDate } = query;
    const start = startDate ? new Date(startDate) : undefined;
    const end = endDate ? new Date(endDate) : undefined;

    return this.analyticsService.getFocusAnalytics(req.user.id, start, end);
  }

  @Get('wellness')
  @ApiOperation({ summary: 'Get wellness analytics for the current user' })
  @ApiResponse({
    status: 200,
    description: 'Wellness analytics retrieved successfully',
    type: WellnessAnalyticsDto
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getWellnessAnalytics(
    @Request() req,
    @Query() query: AnalyticsQueryDto,
  ) {
    const { startDate, endDate } = query;
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
  @ApiResponse({
    status: 200,
    description: 'Team analytics retrieved successfully',
    type: TeamAnalyticsDto
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
    @Query() query: AnalyticsQueryDto,
  ) {
    const { startDate, endDate } = query;
    const start = startDate ? new Date(startDate) : undefined;
    const end = endDate ? new Date(endDate) : undefined;

    return this.analyticsService.getTeamAnalytics(teamId, start, end, req.user.id);
  }
}