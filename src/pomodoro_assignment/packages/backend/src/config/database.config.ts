import { Injectable } from '@nestjs/common';
import { PrismaClient } from '@prisma/client';

@Injectable()
export class DatabaseService extends PrismaClient {
  constructor() {
    super({
      datasources: {
        db: {
          url: process.env.DATABASE_URL,
        },
      },
      log: process.env.NODE_ENV === 'development' ? ['query', 'error'] : ['error'],
      errorFormat: 'pretty',
    });
  }

  async onModuleInit() {
    await this.$connect();

    // SQLite optimizations for better performance
    if (process.env.NODE_ENV === 'production') {
      await this.$executeRaw`PRAGMA journal_mode = WAL;`;
      await this.$executeRaw`PRAGMA synchronous = NORMAL;`;
      await this.$executeRaw`PRAGMA cache_size = 10000;`;
      await this.$executeRaw`PRAGMA temp_store = memory;`;
      await this.$executeRaw`PRAGMA mmap_size = 268435456;`; // 256MB
    }
  }

  async onModuleDestroy() {
    await this.$disconnect();
  }

  async checkHealth(): Promise<{ status: string; responseTime: number }> {
    const start = Date.now();

    try {
      await this.$queryRaw`SELECT 1`;
      const responseTime = Date.now() - start;

      return {
        status: 'healthy',
        responseTime,
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - start,
      };
    }
  }
}