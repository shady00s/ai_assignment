import { Injectable, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { PrismaClient } from '@prisma/client';

@Injectable()
export class PrismaService extends PrismaClient implements OnModuleInit, OnModuleDestroy {
  async onModuleInit() {
    await this.$connect();
    console.log('🗄️ Database connected successfully');
  }

  async onModuleDestroy() {
    await this.$disconnect();
    console.log('🗄️ Database disconnected');
  }

  async cleanDb() {
    await this.task.deleteMany();
    console.log('🧹 Database cleaned');
  }

  async getDatabaseStats() {
    const stats = await this.$queryRaw`SELECT COUNT(*) as total FROM tasks`;
    console.log('📊 Database stats:', stats);
    return stats;
  }
}