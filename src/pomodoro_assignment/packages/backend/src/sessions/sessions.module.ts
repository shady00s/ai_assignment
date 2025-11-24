import { Module } from '@nestjs/common';
import { SessionsController } from './sessions.controller';
import { SessionsService } from './sessions.service';
import { DatabaseService } from '../config/database.config';

@Module({
  controllers: [SessionsController],
  providers: [SessionsService, DatabaseService],
  exports: [SessionsService],
})
export class SessionsModule {}