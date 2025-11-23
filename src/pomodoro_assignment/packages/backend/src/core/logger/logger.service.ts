import { Injectable } from '@nestjs/common';
import * as winston from 'winston';
import { WinstonConfig } from '../../config/winston.config';

@Injectable()
export class LoggerService {
  private logger: winston.Logger;

  constructor() {
    const config = {
      level: process.env.LOG_LEVEL || 'info',
      filePath: process.env.LOG_FILE_PATH || 'logs/app.log',
      errorFilePath: process.env.LOG_ERROR_FILE_PATH || 'logs/error.log',
    };

    this.logger = WinstonConfig.createLogger(config);
  }

  log(message: string, context?: string, meta?: any) {
    this.logger.info(message, { context, ...meta });
  }

  error(message: string, trace?: string, context?: string, meta?: any) {
    this.logger.error(message, { context, trace, ...meta });
  }

  warn(message: string, context?: string, meta?: any) {
    this.logger.warn(message, { context, ...meta });
  }

  debug(message: string, context?: string, meta?: any) {
    this.logger.debug(message, { context, ...meta });
  }

  verbose(message: string, context?: string, meta?: any) {
    this.logger.verbose(message, { context, ...meta });
  }

  // Custom methods for structured logging
  logUserAction(action: string, userId: string, details?: any) {
    this.logger.info(`User action: ${action}`, {
      type: 'USER_ACTION',
      userId,
      action,
      details,
    });
  }

  logApiRequest(req: any, res: any, duration: number) {
    const { method, originalUrl, ip } = req;

    this.logger.info('API Request completed', {
      type: 'API_REQUEST',
      method,
      url: originalUrl,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      ip,
      userAgent: req.headers['user-agent'],
    });
  }

  logError(error: Error, context?: string) {
    this.logger.error('Application error', {
      type: 'APPLICATION_ERROR',
      context,
      message: error.message,
      stack: error.stack,
      name: error.name,
    });
  }

  logPerformance(operation: string, duration: number, details?: any) {
    this.logger.info('Performance metric', {
      type: 'PERFORMANCE',
      operation,
      duration: `${duration}ms`,
      details,
    });
  }

  logSecurity(event: string, details?: any) {
    this.logger.warn('Security event', {
      type: 'SECURITY',
      event,
      details,
      timestamp: new Date().toISOString(),
    });
  }

  // Get the underlying Winston logger for advanced usage
  getLogger(): winston.Logger {
    return this.logger;
  }

  // Create child logger with additional context
  child(context: string): winston.Logger {
    return this.logger.child({ context });
  }
}