import { Module } from '@nestjs/common';
import { WellnessController } from './wellness.controller';
import { WellnessService } from './wellness.service';
import { DatabaseService } from '../config/database.config';

@Module({
  controllers: [WellnessController],
  providers: [WellnessService, DatabaseService],
  exports: [WellnessService],
})
export class WellnessModule {}