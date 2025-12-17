import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { CqrsTaskModule } from './tasks/cqrs-task.module';

@Module({
  imports: [CqrsTaskModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
