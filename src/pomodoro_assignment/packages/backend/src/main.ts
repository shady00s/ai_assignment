
import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { ConfigService } from '@nestjs/config';
import { LoggerService } from './core/logger/logger.service';
import { CoreModule } from './core.module';
import { WinstonConfig } from './config/winston.config';

async function bootstrap() {
  try {
    console.log('Starting OptoPomodoro Backend...');

    const app = await NestFactory.create(CoreModule, {
      logger: ['error', 'warn'], // Use simple logger for debugging
    });

    // Get services
    const configService = app.get(ConfigService);
     const loggerService = app.get(LoggerService);

    // Global prefix
    app.setGlobalPrefix('api');

    // CORS configuration
    app.enableCors({
      origin: configService.get('app.corsOrigin'),
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
    });

    // Global validation pipe
    app.useGlobalPipes(
      new ValidationPipe({
        transform: true,
        whitelist: true,
        forbidNonWhitelisted: true,
        transformOptions: {
          enableImplicitConversion: true,
        },
      }),
    );

    // Global exception handler
    // app.useGlobalInterceptors(app.get(LoggerService)); // Will be implemented later

    // API Documentation (Swagger)
    if (configService.get('app.nodeEnv') !== 'production') {
      const config = new DocumentBuilder()
        .setTitle('OptoPomodoro API')
        .setDescription('Production-ready Pomodoro and task management API')
        .setVersion('1.0.0')
       
        .addBearerAuth()
        .build();

      const document = SwaggerModule.createDocument(app, config);
      SwaggerModule.setup('api/docs', app, document, {
        customSiteTitle: 'OptoPomodoro API Documentation',
        customCss: '.topbar { display: none }',
        customfavIcon: '/favicon.ico',
      });

      console.log('Swagger documentation available at /api/docs');
    }

    // Get port from config
    const port = configService.get<number>('app.port') || 3001;

    // Start server
    await app.listen(port);

    console.log(
      `🚀 OptoPomodoro Backend is running on port ${port}`,
    );

    // Graceful shutdown
    process.on('SIGTERM', async () => {
      loggerService.log('SIGTERM signal received. Starting graceful shutdown...', 'Bootstrap');
      await app.close();
      loggerService.log('Application closed gracefully');
      process.exit(0);
    });

    process.on('SIGINT', async () => {
      loggerService.log('SIGINT signal received. Starting graceful shutdown...', 'Bootstrap');
      await app.close();
      loggerService.log('Application closed gracefully');
      process.exit(0);
    });

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      loggerService.logError(error, 'Uncaught Exception');
       process.exit(1);
    });

    process.on('unhandledRejection', (reason, promise) => {
      loggerService.error('Unhandled Rejection', reason.toString(), 'Unhandled Rejection', { promise });
       process.exit(1);
    });

  } catch (error) {
     console.error('Failed to start application:', error);
    process.exit(1);
  }
}

// Start the application
bootstrap();