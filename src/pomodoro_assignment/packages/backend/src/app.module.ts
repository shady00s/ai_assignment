import { Module, MiddlewareConsumer, NestModule } from '@nestjs/common';
import { APP_GUARD, APP_INTERCEPTOR } from '@nestjs/core';
import { CoreModule } from './core.module';
import { HealthModule } from './core/health/health.module';
import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { TasksModule } from './tasks/tasks.module';
import { SessionsModule } from './sessions/sessions.module';
import { AnalyticsModule } from './analytics/analytics.module';
import { NotificationsModule } from './notifications/notifications.module';
import { AchievementsModule } from './achievements/achievements.module';
import { TeamsModule } from './teams/teams.module';
import { WellnessModule } from './wellness/wellness.module';
import { LoggerService } from './core/logger/logger.service';
import { AuthGuard } from './auth/guards/auth.guard';

// This is the root module that imports the core functionality
// Additional feature modules will be imported here as they are created

@Module({
  imports: [
    CoreModule,
    HealthModule,
    AuthModule,
    UsersModule,
    TasksModule,
    SessionsModule,
    AnalyticsModule,
    NotificationsModule,
    AchievementsModule,
    TeamsModule,
    WellnessModule,
    // Future modules will be added here:
    // ChallengesModule,
    // NotificationsGateway,
  ],
  providers: [
    // Temporarily disable global auth guard for testing
    // {
    //   provide: APP_GUARD,
    //   useClass: AuthGuard,
    // },
  ],
})
export class AppModule {
  configure(consumer: MiddlewareConsumer) {
    // Add any middleware here if needed
  }
}