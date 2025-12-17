import { Module } from '@nestjs/common';
import { CqrsModule } from '@nestjs/cqrs';
import { CqrsTaskController } from './cqrs-task.controller';
import { CreateTaskHandler } from './application/handlers/create-task.handler';
import { GetTasksHandler } from './application/handlers/get-tasks.handler';
import { PrismaTaskRepository } from './infrastructure/prisma-task.repository';
import { PrismaService } from '../prisma.service';

@Module({
  imports: [CqrsModule],
  controllers: [CqrsTaskController],
  providers: [
    PrismaService,
    {
      provide: 'TaskRepository',
      useClass: PrismaTaskRepository,
    },
    CreateTaskHandler,
    GetTasksHandler,
  ],
})
export class CqrsTaskModule {}