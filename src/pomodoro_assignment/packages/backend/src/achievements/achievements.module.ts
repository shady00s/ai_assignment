import { Module } from '@nestjs/common';
import { AchievementsService } from './achievements.service';
import { AchievementsController } from './achievements.controller';
import { DatabaseService } from '../config/database.config';

@Module({
  controllers: [AchievementsController],
  providers: [AchievementsService, DatabaseService],
  exports: [AchievementsService],
})
export class AchievementsModule {}