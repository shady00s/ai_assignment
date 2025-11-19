import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { TasksController } from './tasks.controller';
import { TasksService } from './tasks.service';
import { TasksRepository } from './models/tasks.repository';
import { PrismaTasksRepository } from './models/prisma-tasks.repository';
import { PrismaService } from '../prisma/prisma.service';

const useJsonStorage = process.env.USE_JSON_STORAGE === 'true';

@Module({
  imports: [ConfigModule],
  controllers: [TasksController],
  providers: [
    TasksService,
    TasksRepository,
    PrismaTasksRepository,
    PrismaService,
    {
      provide: 'USE_JSON_STORAGE',
      useValue: useJsonStorage,
    },
  ],
  exports: [TasksService, PrismaService],
})
export class TasksModule {}