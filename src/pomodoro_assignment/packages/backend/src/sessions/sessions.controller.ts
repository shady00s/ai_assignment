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
import { SessionsService, SessionType } from './sessions.service';
import { CreateSessionDto } from './dto/create-session.dto';
import { UpdateSessionDto } from './dto/update-session.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Controller('sessions')
@UseGuards(JwtAuthGuard)
export class SessionsController {
  constructor(private readonly sessionsService: SessionsService) {}

  @Post()
  create(@Body() createSessionDto: CreateSessionDto, @Request() req) {
    return this.sessionsService.createSession(createSessionDto, req.user.id);
  }

  @Get()
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