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
  UseInterceptors,
} from '@nestjs/common';
import { TasksService } from './tasks.service';
import { CreateTaskDto } from './dto/create-task.dto';
import { UpdateTaskDto } from './dto/update-task.dto';
import { TaskResponseDto } from './dto/task-response.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { TaskStatus, Priority } from './dto';
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

@ApiTags('tasks')
@Controller('tasks')
@UseGuards(JwtAuthGuard)
export class TasksController {
  constructor(private readonly tasksService: TasksService) {}

  @Post()
  @ApiOperation({ summary: 'Create a new task' })
  @ApiStandardResponse(TaskResponseDto, 'Task created successfully')
  create(@Body() createTaskDto: CreateTaskDto, @Request() req) {
    return this.tasksService.create(createTaskDto, req.user.id);
  }

  @Get()
  @ApiOperation({ summary: 'Get all tasks for the authenticated user' })
  @ApiQuery({ name: 'status', required: false, description: 'Filter by task status', isArray: true })
  @ApiQuery({ name: 'priority', required: false, description: 'Filter by task priority', isArray: true })
  @ApiQuery({ name: 'teamId', required: false, description: 'Filter by team ID' })
  @ApiQuery({ name: 'assigneeId', required: false, description: 'Filter by assignee ID' })
  @ApiQuery({ name: 'tags', required: false, description: 'Filter by tags', isArray: true })
  @ApiQuery({ name: 'sortBy', required: false, description: 'Sort field', enum: ['createdAt', 'updatedAt', 'dueDate', 'priority', 'title', 'status'] })
  @ApiQuery({ name: 'sortOrder', required: false, description: 'Sort order', enum: ['ASC', 'DESC'] })
  @ApiStandardArrayResponse(TaskResponseDto, 'Tasks retrieved successfully')
  findAll(
    @Request() req,
    @Query('status') status?: string | string[],
    @Query('priority') priority?: string | string[],
    @Query('teamId') teamId?: string,
    @Query('assigneeId') assigneeId?: string,
    @Query('tags') tags?: string | string[],
    @Query('sortBy') sortBy?: string,
    @Query('sortOrder') sortOrder?: 'ASC' | 'DESC',
  ) {
    const filters: any = {
      ...(status && { status }),
      ...(priority && { priority }),
      ...(teamId && { teamId }),
      ...(assigneeId && { assigneeId }),
      ...(tags && { tags }),
    };

    // Handle sorting
    const sort = sortBy && sortOrder ? { field: sortBy, direction: sortOrder } : {};

    return this.tasksService.findAll(req.user.id, filters);
  }

  @Get('analytics')
  getTaskAnalytics(
    @Request() req,
    @Query('startDate') startDate?: string,
    @Query('endDate') endDate?: string,
  ) {
    const start = startDate ? new Date(startDate) : undefined;
    const end = endDate ? new Date(endDate) : undefined;

    return this.tasksService.getTaskAnalytics(req.user.id, start, end);
  }

  @Get('project/:projectId')
  getTasksByProject(@Param('projectId') projectId: string, @Request() req) {
    return this.tasksService.getTasksByProject(req.user.id, projectId);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a specific task by ID' })
  @ApiParam({ name: 'id', description: 'Task ID' })
  @ApiStandardResponse(TaskResponseDto, 'Task retrieved successfully')
  findOne(@Param('id') id: string, @Request() req) {
    return this.tasksService.findOne(id, req.user.id);
  }

  @Patch(':id')
  @ApiOperation({ summary: 'Update a task' })
  @ApiParam({ name: 'id', description: 'Task ID' })
  @ApiStandardResponse(TaskResponseDto, 'Task updated successfully')
  update(@Param('id') id: string, @Body() updateTaskDto: UpdateTaskDto, @Request() req) {
    return this.tasksService.update(id, updateTaskDto, req.user.id);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Delete a task' })
  @ApiParam({ name: 'id', description: 'Task ID' })
  @ApiDeleteResponse('Task deleted successfully')
  remove(@Param('id') id: string, @Request() req) {
    return this.tasksService.remove(id, req.user.id);
  }

  @Post(':id/dependencies')
  addDependency(
    @Param('id') taskId: string,
    @Body() body: { prerequisiteId: string },
    @Request() req,
  ) {
    return this.tasksService.addDependency(taskId, body.prerequisiteId, req.user.id);
  }

  @Delete(':id/dependencies/:prerequisiteId')
  removeDependency(
    @Param('id') taskId: string,
    @Param('prerequisiteId') prerequisiteId: string,
    @Request() req,
  ) {
    return this.tasksService.removeDependency(taskId, prerequisiteId, req.user.id);
  }
}