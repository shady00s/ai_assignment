import { forwardRef, Module } from '@nestjs/common';
import { TasksController } from './tasks.controller';
import { TasksService } from './tasks.service';
import { CoreModule } from '@/core.module';
import { UsersService } from '@/users/users.service';

@Module({
  imports: [
    forwardRef(()=>CoreModule)
    
  ],
  controllers: [TasksController],
  providers: [TasksService, UsersService],
  exports: [TasksService],
})
export class TasksModule {}