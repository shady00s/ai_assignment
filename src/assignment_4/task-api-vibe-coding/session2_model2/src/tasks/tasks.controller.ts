import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  Query,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiParam } from '@nestjs/swagger';
import { TasksService } from './tasks.service';
import { CreateTaskDto } from './dto/create-task.dto';
import { UpdateTaskDto } from './dto/update-task.dto';
import { TaskFilterDto } from './dto/task-filter.dto';
import { TaskResponseDto } from './dto/task-response.dto';

@ApiTags('tasks')
@Controller('tasks')
export class TasksController {
  constructor(private readonly tasksService: TasksService) {}

  @Post()
  @ApiOperation({
    summary: 'Create a new task',
    description: 'Creates a new task with the provided title, optional description, and priority. Tasks are created with PENDING status by default.'
  })
  @ApiResponse({
    status: 201,
    description: 'Task created successfully',
    type: TaskResponseDto
  })
  @ApiResponse({
    status: 400,
    description: 'Bad request - Invalid input data'
  })
  async create(@Body() createTaskDto: CreateTaskDto): Promise<TaskResponseDto> {
    return this.tasksService.create(createTaskDto);
  }

  @Get()
  @ApiOperation({
    summary: 'Get all tasks',
    description: 'Retrieves a list of all tasks. Optional filters can be applied to filter by status and/or priority. Results are ordered by creation date (newest first).'
  })
  @ApiResponse({
    status: 200,
    description: 'Tasks retrieved successfully',
    type: [TaskResponseDto]
  })
  async findAll(@Query() filterDto: TaskFilterDto): Promise<TaskResponseDto[]> {
    return this.tasksService.findAll(filterDto);
  }

  @Get(':id')
  @ApiOperation({
    summary: 'Get task by ID',
    description: 'Retrieves a specific task by its unique identifier.'
  })
  @ApiParam({
    name: 'id',
    description: 'Unique identifier of the task',
    example: 'cuy1t2y3k0000l2z8s0r1x2a4'
  })
  @ApiResponse({
    status: 200,
    description: 'Task retrieved successfully',
    type: TaskResponseDto
  })
  @ApiResponse({
    status: 404,
    description: 'Task not found'
  })
  async findOne(@Param('id') id: string): Promise<TaskResponseDto> {
    return this.tasksService.findOne(id);
  }

  @Patch(':id')
  @ApiOperation({
    summary: 'Update a task',
    description: 'Updates specific fields of an existing task. Only provided fields will be updated; others remain unchanged.'
  })
  @ApiParam({
    name: 'id',
    description: 'Unique identifier of the task to update',
    example: 'cuy1t2y3k0000l2z8s0r1x2a4'
  })
  @ApiResponse({
    status: 200,
    description: 'Task updated successfully',
    type: TaskResponseDto
  })
  @ApiResponse({
    status: 404,
    description: 'Task not found'
  })
  @ApiResponse({
    status: 400,
    description: 'Bad request - Invalid input data'
  })
  async update(
    @Param('id') id: string,
    @Body() updateTaskDto: UpdateTaskDto,
  ): Promise<TaskResponseDto> {
    return this.tasksService.update(id, updateTaskDto);
  }

  @Delete(':id')
  @ApiOperation({
    summary: 'Delete a task',
    description: 'Permanently deletes a task by its unique identifier. This action cannot be undone.'
  })
  @ApiParam({
    name: 'id',
    description: 'Unique identifier of the task to delete',
    example: 'cuy1t2y3k0000l2z8s0r1x2a4'
  })
  @ApiResponse({
    status: 204,
    description: 'Task deleted successfully'
  })
  @ApiResponse({
    status: 404,
    description: 'Task not found'
  })
  @HttpCode(HttpStatus.NO_CONTENT)
  async remove(@Param('id') id: string): Promise<void> {
    return this.tasksService.remove(id);
  }
}