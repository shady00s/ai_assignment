import { Module, forwardRef } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { ThrottlerModule } from '@nestjs/throttler';
import configuration from './config/configuration';
import { validate } from './config/validation';
import { DatabaseService } from './config/database.config';
import { LoggerModule } from './core/logger/logger.module';
import { HealthModule } from './core/health/health.module';
import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { TasksModule } from './tasks/tasks.module';

@Module({
  imports: [
    // Configuration
    ConfigModule.forRoot({
      isGlobal: true,
      load: [configuration],
      // validate, // Temporarily disabled for debugging
      envFilePath: ['.env.local', '.env', '.env.production'],
    }),

    // Rate limiting
    ThrottlerModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: (configService: ConfigService) => ({
        throttlers: [
          {
            ttl: configService.get('app.rateLimit.ttl') * 1000,
            limit: configService.get('app.rateLimit.max'),
          },
        ],
      }),
      inject: [ConfigService],
    }),

    // Core modules
    LoggerModule,
    HealthModule,
    forwardRef(() => AuthModule),
    forwardRef(() => UsersModule),
    forwardRef(() => TasksModule),
   ],
  providers: [DatabaseService],
  exports: [DatabaseService],
})
export class CoreModule {}