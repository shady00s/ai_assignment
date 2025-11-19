import {
  Controller,
  Get,
  Post,
  Body,
  Put,
  Param,
  Delete,
  Query,
  HttpCode,
  HttpStatus
} from '@nestjs/common';
import { TasksService } from './tasks.service';
import { CreateTaskDto } from './dto/create-task.dto';
import { UpdateTaskDto } from './dto/update-task.dto';
import { QueryTasksDto } from './dto/query-tasks.dto';
import { Task, TaskStatus } from './entities/task.entity';

@Controller('tasks')
export class TasksController {
  constructor(private readonly tasksService: TasksService) {}

  @Get()
  async findAll(@Query() query: QueryTasksDto): Promise<Task[]> {
    return await this.tasksService.findAll(query);
  }

  @Get('stats')
  async getStats() {
    return await this.tasksService.getStats();
  }

  @Get(':id')
  async findOne(@Param('id') id: string): Promise<Task> {
    return await this.tasksService.findOne(id);
  }

  @Post()
  @HttpCode(HttpStatus.CREATED)
  async create(@Body() createTaskDto: CreateTaskDto): Promise<Task> {
    return await this.tasksService.create(createTaskDto);
  }

  @Put(':id')
  async update(
    @Param('id') id: string,
    @Body() updateTaskDto: UpdateTaskDto
  ): Promise<Task> {
    return await this.tasksService.update(id, updateTaskDto);
  }

  @Delete(':id')
  @HttpCode(HttpStatus.NO_CONTENT)
  async remove(@Param('id') id: string): Promise<void> {
    await this.tasksService.remove(id);
  }

  @Get('storage-backend')
  getStorageBackend() {
    return {
      backend: this.tasksService.getStorageBackend(),
      message: `Currently using ${this.tasksService.getStorageBackend()} storage backend`,
    };
  }

  @Post('bulk')
  @HttpCode(HttpStatus.CREATED)
  async bulkCreate(@Body() createTaskDtos: CreateTaskDto[]): Promise<Task[]> {
    return await this.tasksService.bulkCreate(createTaskDtos);
  }

  @Put('bulk/status')
  async bulkUpdateStatus(
    @Body() body: { ids: string[]; status: TaskStatus }
  ): Promise<{ count: number }> {
    return await this.tasksService.bulkUpdateStatus(body.ids, body.status);
  }
}