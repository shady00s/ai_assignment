import { PrismaClient } from '@prisma/client';
import * as fs from 'fs/promises';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';

const prisma = new PrismaClient();

async function migrateFromJson() {
  try {
    console.log('🔄 Starting data migration from JSON to SQLite...');

    // Read existing JSON data
    const jsonFilePath = path.join(process.cwd(), 'data', 'tasks.json');

    try {
      const data = await fs.readFile(jsonFilePath, 'utf-8');
      const taskData = JSON.parse(data);

      if (taskData.tasks && taskData.tasks.length > 0) {
        console.log(`📋 Found ${taskData.tasks.length} tasks in JSON file`);

        // Clear existing database data
        await prisma.task.deleteMany();
        console.log('🧹 Cleared existing database data');

        // Migrate each task to database
        let migratedCount = 0;
        for (const task of taskData.tasks) {
          try {
            await prisma.task.create({
              data: {
                id: task.id || uuidv4(),
                title: task.title,
                description: task.description || null,
                status: task.status === 'completed' ? 'COMPLETED' : 'PENDING',
                createdAt: new Date(task.created_at),
                updatedAt: new Date(task.updated_at),
              },
            });
            migratedCount++;
          } catch (error) {
            console.warn(`⚠️  Failed to migrate task: ${task.title}`, error);
          }
        }

        console.log(`✅ Successfully migrated ${migratedCount} tasks to SQLite database`);

        // Verify migration
        const totalTasks = await prisma.task.count();
        console.log(`📊 Database now contains ${totalTasks} tasks`);
      } else {
        console.log('ℹ️  No tasks found in JSON file, skipping migration');
      }
    } catch (error) {
      console.log('ℹ️  JSON file not found, starting with empty database');
    }

    console.log('🎉 Migration completed successfully!');
  } catch (error) {
    console.error('❌ Migration failed:', error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

// Export the function for testing
export { migrateFromJson };

// Run migration if called directly
if (require.main === module) {
  migrateFromJson();
}