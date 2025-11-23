import { plainToClass } from 'class-transformer';
import { validateSync } from 'class-validator';

class EnvironmentVariables {
  PORT: number;
  NODE_ENV: string;
  DATABASE_URL: string;

  JWT_SECRET: string;
  JWT_REFRESH_SECRET: string;
  JWT_EXPIRATION: string;
  JWT_REFRESH_EXPIRATION: string;

  CORS_ORIGIN: string;

  LOG_LEVEL: string;
  LOG_FILE_PATH: string;
  LOG_ERROR_FILE_PATH: string;

  RATE_LIMIT_TTL: number;
  RATE_LIMIT_MAX: number;
}

export function validate(config: Record<string, unknown>) {
  const validatedConfig = plainToClass(EnvironmentVariables, config, {
    enableImplicitConversion: true,
  });

  const errors = validateSync(validatedConfig, {
    skipMissingProperties: false,
  });

  if (errors.length > 0) {
    throw new Error(errors.map(error => Object.values(error.constraints!).join(', ')).join('\n'));
  }

  return validatedConfig;
}

export const configurationValidation = {
  isString: (value: any): boolean => typeof value === 'string' && value.trim().length > 0,
  isNumber: (value: any): boolean => typeof value === 'number' && !isNaN(value),
  isBoolean: (value: any): boolean => typeof value === 'boolean',
  isUrl: (value: any): boolean => {
    if (typeof value !== 'string') return false;
    try {
      new URL(value);
      return true;
    } catch {
      return false;
    }
  },
  isEmail: (value: any): boolean => {
    if (typeof value !== 'string') return false;
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(value);
  },
  isJwtSecret: (value: any): boolean => {
    if (typeof value !== 'string') return false;
    return value.length >= 32; // Minimum 32 characters for security
  },
  isLogLevel: (value: any): boolean => {
    const validLevels = ['error', 'warn', 'info', 'debug', 'verbose'];
    return typeof value === 'string' && validLevels.includes(value);
  },
};