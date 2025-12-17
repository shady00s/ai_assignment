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
import { CommandBus, QueryBus } from '@nestjs/cqrs';
import { ApiTags, ApiOperation, ApiResponse, ApiParam } from '@nestjs/swagger';
import { CreateTaskCommand } from './application/commands/create-task.command';
import { UpdateTaskCommand } from './application/commands/update-task.command';
import { DeleteTaskCommand } from './application/commands/delete-task.command';
import { GetTaskByIdQuery } from './application/queries/get-task-by-id.query';
import { GetTasksQuery } from './application/queries/get-tasks.query';
import { TaskResponseDto } from './application/dtos/task-response.dto';
import { CreateTaskRequestDto } from './application/dtos/create-task-request.dto';
import { UpdateTaskRequestDto } from './application/dtos/update-task-request.dto';
import { TaskFilterDto } from './application/dtos/task-filter.dto';

@ApiTags('tasks')
@Controller('tasks')
export class CqrsTaskController {
  constructor(
    private readonly commandBus: CommandBus,
    private readonly queryBus: QueryBus
  ) {}

  @Post()
  @ApiOperation({
    summary: 'Create a new task (CQRS)',
    description: 'Creates a new task using Command Query Responsibility Segregation pattern. Tasks are created with PENDING status by default.'
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
  async create(@Body() createTaskDto: CreateTaskRequestDto): Promise<TaskResponseDto> {
    const command = CreateTaskCommand.create(
      createTaskDto.title,
      createTaskDto.description,
      createTaskDto.priority
    );

    await this.commandBus.execute(command);

    // After creation, retrieve the task to return it
    // In a real application, you might return the task ID or use a different approach
    const query = GetTasksQuery.create();
    const tasks = await this.queryBus.execute(query);
    return tasks[0]; // Return the most recently created task
  }

  @Get()
  @ApiOperation({
    summary: 'Get all tasks (CQRS)',
    description: 'Retrieves tasks using CQRS pattern with read-side optimization. Optional filters can be applied to filter by status and/or priority.'
  })
  @ApiResponse({
    status: 200,
    description: 'Tasks retrieved successfully',
    type: [TaskResponseDto]
  })
  async findAll(@Query() filterDto: TaskFilterDto): Promise<TaskResponseDto[]> {
    const query = GetTasksQuery.create({
      status: filterDto.status,
      priority: filterDto.priority,
    });

    return this.queryBus.execute(query);
  }

  @Get(':id')
  @ApiOperation({
    summary: 'Get task by ID (CQRS)',
    description: 'Retrieves a specific task by its unique identifier using the query side of CQRS.'
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
    const query = GetTaskByIdQuery.create(id);
    return this.queryBus.execute(query);
  }

  @Patch(':id')
  @ApiOperation({
    summary: 'Update a task (CQRS)',
    description: 'Updates specific fields of an existing task using the command side of CQRS. Only provided fields will be updated.'
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
  async update(
    @Param('id') id: string,
    @Body() updateTaskDto: UpdateTaskRequestDto
  ): Promise<TaskResponseDto> {
    const command = UpdateTaskCommand.create(id, updateTaskDto);

    await this.commandBus.execute(command);

    // Return updated task
    const query = GetTaskByIdQuery.create(id);
    return this.queryBus.execute(query);
  }

  @Delete(':id')
  @ApiOperation({
    summary: 'Delete a task (CQRS)',
    description: 'Permanently deletes a task by its unique identifier using the command side of CQRS. This action cannot be undone.'
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
    const command = DeleteTaskCommand.create(id);
    await this.commandBus.execute(command);
  }
}