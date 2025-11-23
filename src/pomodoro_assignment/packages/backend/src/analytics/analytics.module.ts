import { Module } from '@nestjs/common';
import { AnalyticsController } from './analytics.controller';
import { AnalyticsService } from './analytics.service';
import { DatabaseService } from '../config/database.config';

@Module({
  controllers: [AnalyticsController],
  providers: [AnalyticsService, DatabaseService],
  exports: [AnalyticsService],
})
export class AnalyticsModule {}