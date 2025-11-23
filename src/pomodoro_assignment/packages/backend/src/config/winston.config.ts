import * as winston from 'winston';
import * as path from 'path';

export class WinstonConfig {
  static createLogger(config: { level: string; filePath: string; errorFilePath: string }) {
    const logFormat = winston.format.combine(
      winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
      winston.format.errors({ stack: true }),
      winston.format.json(),
    );

    const consoleFormat = winston.format.combine(
      winston.format.colorize(),
      winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
      winston.format.printf(({ timestamp, level, message, stack, ...meta }) => {
        let msg = `${timestamp} [${level}]: ${message}`;

        if (stack) {
          msg += `\n${stack}`;
        }

        if (Object.keys(meta).length > 0) {
          msg += `\n${JSON.stringify(meta, null, 2)}`;
        }

        return msg;
      }),
    );

    // Ensure logs directory exists
    const logDir = path.dirname(config.filePath);
    require('fs').mkdirSync(logDir, { recursive: true });

    return winston.createLogger({
      level: config.level,
      format: logFormat,
      defaultMeta: { service: 'optopomodoro-backend' },
      transports: [
        // Console transport for development
        new winston.transports.Console({
          format: consoleFormat,
        }),

        // File transport for all logs
        new winston.transports.File({
          filename: config.filePath,
          maxsize: 5242880, // 5MB
          maxFiles: 10,
          tailable: true,
        }),

        // Separate file transport for errors
        new winston.transports.File({
          filename: config.errorFilePath,
          level: 'error',
          maxsize: 5242880, // 5MB
          maxFiles: 5,
          tailable: true,
        }),
      ],

      // Handle uncaught exceptions and rejections
      exceptionHandlers: [
        new winston.transports.File({
          filename: path.join(logDir, 'exceptions.log'),
        }),
      ],

      rejectionHandlers: [
        new winston.transports.File({
          filename: path.join(logDir, 'rejections.log'),
        }),
      ],

      // Exit on error in production
      exitOnError: config.level === 'error',
    });
  }
}