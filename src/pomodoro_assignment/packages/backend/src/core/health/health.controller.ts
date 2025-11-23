import { Controller, Get, Inject } from '@nestjs/common';
import { ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import { DatabaseService } from '../../config/database.config';
import { LoggerService } from '../logger/logger.service';

export interface HealthCheckResult {
  status: 'UP' | 'DOWN';
  timestamp: string;
  services: {
    database: {
      status: 'UP' | 'DOWN';
      responseTime: number;
    };
    memory: {
      status: 'UP' | 'DOWN';
      usage: number;
      total: number;
      percentage: number;
    };
    uptime: {
      status: 'UP' | 'DOWN';
      seconds: number;
      formatted: string;
    };
  };
  version: string;
  environment: string;
}

@ApiTags('Health')
@Controller('health')
export class HealthController {
  constructor(
     private readonly databaseService: DatabaseService,
    private readonly logger: LoggerService,
  ) {}

  @Get()
  @ApiOperation({ summary: 'Check application health' })
  @ApiResponse({
    status: 200,
    description: 'Application is healthy',
    schema: {
      type: 'object',
      properties: {
        status: { type: 'string', enum: ['UP', 'DOWN'] },
        timestamp: { type: 'string' },
        services: {
          type: 'object',
          properties: {
            database: { type: 'object' },
            memory: { type: 'object' },
            uptime: { type: 'object' },
          },
        },
        version: { type: 'string' },
        environment: { type: 'string' },
      },
    },
  })
  async check(): Promise<HealthCheckResult> {
    const startTime = Date.now();

    try {
      // Check database health
      const dbHealth = await this.databaseService.checkHealth();

      // Get memory usage
      const memoryUsage = process.memoryUsage();
      const totalMemory = require('os').totalmem();

      // Calculate uptime
      const uptime = process.uptime();

      const result: HealthCheckResult = {
        status: dbHealth.status === 'healthy' ? 'UP' : 'DOWN',
        timestamp: new Date().toISOString(),
        services: {
          database: {
            status: dbHealth.status === 'healthy' ? 'UP' : 'DOWN',
            responseTime: dbHealth.responseTime,
          },
          memory: {
            status: 'UP', // Always up if we can measure it
            usage: memoryUsage.heapUsed,
            total: totalMemory,
            percentage: Math.round((memoryUsage.heapUsed / totalMemory) * 100 * 100) / 100,
          },
          uptime: {
            status: 'UP', // Always up if we can measure it
            seconds: Math.round(uptime),
            formatted: this.formatUptime(uptime),
          },
        },
        version: process.env.npm_package_version || '1.0.0',
        environment: process.env.NODE_ENV || 'development',
      };

      this.logger.log('Health check completed', 'HealthController', {
        status: result.status,
        responseTime: Date.now() - startTime,
        services: result.services,
      });

      return result;
    } catch (error) {
      const responseTime = Date.now() - startTime;

      this.logger.error('Health check failed', error.stack, 'HealthController', {
        responseTime,
      });

      // Return DOWN status but still include available information
      return {
        status: 'DOWN',
        timestamp: new Date().toISOString(),
        services: {
          database: {
            status: 'DOWN',
            responseTime,
          },
          memory: {
            status: 'UP',
            usage: process.memoryUsage().heapUsed,
            total: require('os').totalmem(),
            percentage: Math.round((process.memoryUsage().heapUsed / require('os').totalmem()) * 100 * 100) / 100,
          },
          uptime: {
            status: 'UP',
            seconds: Math.round(process.uptime()),
            formatted: this.formatUptime(process.uptime()),
          },
        },
        version: process.env.npm_package_version || '1.0.0',
        environment: process.env.NODE_ENV || 'development',
      };
    }
  }

  @Get('readiness')
  @ApiOperation({ summary: 'Check if application is ready to serve traffic' })
  @ApiResponse({ status: 200, description: 'Application is ready' })
  async readiness() {
    try {
      // Check critical dependencies
      await this.databaseService.checkHealth();

      return {
        status: 'READY',
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      this.logger.error('Readiness check failed', error.stack, 'HealthController');

      return {
        status: 'NOT_READY',
        timestamp: new Date().toISOString(),
        error: error.message,
      };
    }
  }

  @Get('liveness')
  @ApiOperation({ summary: 'Check if application is alive' })
  @ApiResponse({ status: 200, description: 'Application is alive' })
  async liveness() {
    // Basic liveness check - if we can respond, we're alive
    return {
      status: 'ALIVE',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
    };
  }

  private formatUptime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    const parts = [];
    if (hours > 0) parts.push(`${hours}h`);
    if (minutes > 0) parts.push(`${minutes}m`);
    if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);

    return parts.join(' ');
  }
}