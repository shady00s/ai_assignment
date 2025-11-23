import { Module } from '@nestjs/common';
import { HealthController } from './health.controller';
import { DatabaseService } from '../../config/database.config';
import { LoggerModule } from '../logger/logger.module';

@Module({
  imports: [LoggerModule],
  controllers: [HealthController],
  providers: [DatabaseService],
})
export class HealthModule {}