import { Module } from '@nestjs/common';
import { TeamsController } from './teams.controller';
import { TeamsService } from './teams.service';
import { DatabaseService } from '../config/database.config';

@Module({
  controllers: [TeamsController],
  providers: [TeamsService, DatabaseService],
  exports: [TeamsService],
})
export class TeamsModule {}