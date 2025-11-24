import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  Query,
  UseGuards,
  Request,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiParam, ApiQuery } from '@nestjs/swagger';
import { SessionsService, SessionType } from './sessions.service';
import { CreateSessionDto } from './dto/create-session.dto';
import { UpdateSessionDto } from './dto/update-session.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('sessions')
@Controller('sessions')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class SessionsController {
  constructor(private readonly sessionsService: SessionsService) {}

  @Post()
  create(@Body() createSessionDto: CreateSessionDto, @Request() req) {
    return this.sessionsService.createSession(createSessionDto, req.user.id);
  }

  @Get()
  @ApiOperation({ summary: 'Get all sessions for the current user' })
  @ApiResponse({
    status: 200,
    description: 'Sessions retrieved successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiQuery({ name: 'type', required: false, description: 'Filter by session type' })
  @ApiQuery({ name: 'taskId', required: false, description: 'Filter by task ID' })
  @ApiQuery({ name: 'startDate', required: false, description: 'Filter by start date' })
  @ApiQuery({ name: 'endDate', required: false, description: 'Filter by end date' })
  findAll(
    @Request() req,
    @Query('type') type?: string,
    @Query('taskId') taskId?: string,
    @Query('startDate') startDate?: string,
    @Query('endDate') endDate?: string,
  ) {
    const filters = {
      ...(type && { type: type as SessionType }),
      ...(taskId && { taskId }),
      ...(startDate && { startDate: new Date(startDate) }),
      ...(endDate && { endDate: new Date(endDate) }),
    };

    return this.sessionsService.findAll(req.user.id, filters);
  }

  @Get('analytics')
  getSessionAnalytics(
    @Request() req,
    @Query('startDate') startDate?: string,
    @Query('endDate') endDate?: string,
  ) {
    const start = startDate ? new Date(startDate) : undefined;
    const end = endDate ? new Date(endDate) : undefined;

    return this.sessionsService.getSessionAnalytics(req.user.id, start, end);
  }

  @Get('active')
  getActiveSession(@Request() req) {
    return this.sessionsService.getActiveSession(req.user.id);
  }

  @Get(':id')
  findOne(@Param('id') id: string, @Request() req) {
    return this.sessionsService.findOne(id, req.user.id);
  }

  @Patch(':id')
  update(@Param('id') id: string, @Body() updateSessionDto: UpdateSessionDto, @Request() req) {
    return this.sessionsService.update(id, updateSessionDto, req.user.id);
  }

  @Post(':id/complete')
  completeSession(
    @Param('id') id: string,
    @Body() body: { quality?: number; notes?: string },
    @Request() req,
  ) {
    return this.sessionsService.completeSession(id, req.user.id, body.quality, body.notes);
  }

  @Post(':id/start')
  startSession(@Param('id') id: string, @Request() req) {
    return this.sessionsService.startSession(id, req.user.id);
  }

  @Post(':id/pause')
  pauseSession(@Param('id') id: string, @Request() req) {
    return this.sessionsService.pauseSession(id, req.user.id);
  }

  @Delete(':id')
  remove(@Param('id') id: string, @Request() req) {
    return this.sessionsService.remove(id, req.user.id);
  }
}