/*
  Warnings:

  - You are about to drop the `_ChallengeParticipant` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `challenges` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the column `badge_url` on the `achievements` table. All the data in the column will be lost.
  - You are about to drop the column `requirement_timeframe` on the `achievements` table. All the data in the column will be lost.
  - You are about to drop the column `requirement_type` on the `achievements` table. All the data in the column will be lost.
  - You are about to drop the column `requirement_value` on the `achievements` table. All the data in the column will be lost.
  - You are about to drop the column `xp_reward` on the `achievements` table. All the data in the column will be lost.
  - You are about to drop the column `action_data` on the `notifications` table. All the data in the column will be lost.
  - You are about to drop the column `action_label` on the `notifications` table. All the data in the column will be lost.
  - You are about to drop the column `action_url` on the `notifications` table. All the data in the column will be lost.
  - You are about to drop the column `priority` on the `notifications` table. All the data in the column will be lost.
  - You are about to drop the column `read` on the `notifications` table. All the data in the column will be lost.
  - You are about to drop the column `related_entity_id` on the `notifications` table. All the data in the column will be lost.
  - You are about to drop the column `related_entity_type` on the `notifications` table. All the data in the column will be lost.
  - You are about to drop the column `sender_id` on the `notifications` table. All the data in the column will be lost.
  - You are about to drop the column `timestamp` on the `notifications` table. All the data in the column will be lost.
  - You are about to drop the column `completed_pomodoros` on the `tasks` table. All the data in the column will be lost.
  - You are about to drop the column `deleted_at` on the `tasks` table. All the data in the column will be lost.
  - You are about to drop the column `estimated_pomodoros` on the `tasks` table. All the data in the column will be lost.
  - You are about to drop the column `project_id` on the `tasks` table. All the data in the column will be lost.
  - You are about to drop the column `deleted_at` on the `teams` table. All the data in the column will be lost.
  - You are about to drop the column `deleted_at` on the `users` table. All the data in the column will be lost.
  - You are about to drop the column `password_hash` on the `users` table. All the data in the column will be lost.
  - You are about to drop the column `team_id` on the `users` table. All the data in the column will be lost.
  - Added the required column `criteria` to the `achievements` table without a default value. This is not possible if the table is not empty.
  - Added the required column `xpValue` to the `achievements` table without a default value. This is not possible if the table is not empty.
  - Added the required column `creator_id` to the `tasks` table without a default value. This is not possible if the table is not empty.

*/
-- DropIndex
DROP INDEX "_ChallengeParticipant_B_index";

-- DropIndex
DROP INDEX "_ChallengeParticipant_AB_unique";

-- DropIndex
DROP INDEX "challenges_current_value_target_value_idx";

-- DropIndex
DROP INDEX "challenges_team_id_idx";

-- DropIndex
DROP INDEX "challenges_created_by_idx";

-- DropIndex
DROP INDEX "challenges_start_date_end_date_idx";

-- DropIndex
DROP INDEX "challenges_type_idx";

-- DropIndex
DROP INDEX "team_members_role_idx";

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "_ChallengeParticipant";
PRAGMA foreign_keys=on;

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "challenges";
PRAGMA foreign_keys=on;

-- CreateTable
CREATE TABLE "task_dependencies" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "dependent_task_id" TEXT NOT NULL,
    "prerequisite_id" TEXT NOT NULL,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "task_dependencies_dependent_task_id_fkey" FOREIGN KEY ("dependent_task_id") REFERENCES "tasks" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "task_dependencies_prerequisite_id_fkey" FOREIGN KEY ("prerequisite_id") REFERENCES "tasks" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "team_challenges" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "team_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "type" TEXT NOT NULL,
    "target_value" INTEGER NOT NULL,
    "current_value" INTEGER NOT NULL DEFAULT 0,
    "start_date" DATETIME NOT NULL,
    "end_date" DATETIME NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "team_challenges_team_id_fkey" FOREIGN KEY ("team_id") REFERENCES "teams" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_achievements" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "icon" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "xpValue" INTEGER NOT NULL,
    "criteria" TEXT NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "new_achievements" ("category", "created_at", "description", "icon", "id", "name") SELECT "category", "created_at", "description", "icon", "id", "name" FROM "achievements";
DROP TABLE "achievements";
ALTER TABLE "new_achievements" RENAME TO "achievements";
CREATE INDEX "achievements_category_idx" ON "achievements"("category");
CREATE INDEX "achievements_isActive_idx" ON "achievements"("isActive");
CREATE TABLE "new_notifications" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "user_id" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "data" TEXT,
    "read_at" DATETIME,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "notifications_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_notifications" ("id", "message", "title", "type", "user_id") SELECT "id", "message", "title", "type", "user_id" FROM "notifications";
DROP TABLE "notifications";
ALTER TABLE "new_notifications" RENAME TO "notifications";
CREATE INDEX "notifications_user_id_idx" ON "notifications"("user_id");
CREATE INDEX "notifications_read_at_idx" ON "notifications"("read_at");
CREATE INDEX "notifications_created_at_idx" ON "notifications"("created_at");
CREATE TABLE "new_sessions" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "user_id" TEXT NOT NULL,
    "task_id" TEXT,
    "type" TEXT NOT NULL DEFAULT 'POMODORO',
    "duration" INTEGER NOT NULL,
    "started_at" DATETIME NOT NULL,
    "completed_at" DATETIME,
    "notes" TEXT,
    "quality" INTEGER,
    "completed" BOOLEAN NOT NULL DEFAULT false,
    "interruptions" INTEGER NOT NULL DEFAULT 0,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "sessions_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "sessions_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "tasks" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "new_sessions" ("completed", "completed_at", "created_at", "duration", "id", "interruptions", "notes", "quality", "started_at", "task_id", "type", "user_id") SELECT "completed", "completed_at", "created_at", "duration", "id", "interruptions", "notes", "quality", "started_at", "task_id", "type", "user_id" FROM "sessions";
DROP TABLE "sessions";
ALTER TABLE "new_sessions" RENAME TO "sessions";
CREATE INDEX "sessions_user_id_started_at_idx" ON "sessions"("user_id", "started_at");
CREATE INDEX "sessions_user_id_type_idx" ON "sessions"("user_id", "type");
CREATE INDEX "sessions_completed_at_idx" ON "sessions"("completed_at");
CREATE INDEX "sessions_task_id_idx" ON "sessions"("task_id");
CREATE INDEX "sessions_completed_idx" ON "sessions"("completed");
CREATE TABLE "new_tasks" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "title" TEXT NOT NULL,
    "description" TEXT,
    "priority" TEXT NOT NULL DEFAULT 'MEDIUM',
    "status" TEXT NOT NULL DEFAULT 'TODO',
    "due_date" DATETIME,
    "estimatedPomodoros" INTEGER NOT NULL DEFAULT 1,
    "completedPomodoros" INTEGER NOT NULL DEFAULT 0,
    "estimatedMinutes" INTEGER,
    "actualMinutes" INTEGER,
    "assignee_id" TEXT,
    "creator_id" TEXT NOT NULL,
    "team_id" TEXT,
    "tags" TEXT,
    "complexity" INTEGER NOT NULL DEFAULT 1,
    "completed_at" DATETIME,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "tasks_creator_id_fkey" FOREIGN KEY ("creator_id") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "tasks_assignee_id_fkey" FOREIGN KEY ("assignee_id") REFERENCES "users" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "tasks_team_id_fkey" FOREIGN KEY ("team_id") REFERENCES "teams" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "new_tasks" ("assignee_id", "completed_at", "created_at", "description", "due_date", "id", "priority", "status", "tags", "title", "updated_at") SELECT "assignee_id", "completed_at", "created_at", "description", "due_date", "id", "priority", "status", "tags", "title", "updated_at" FROM "tasks";
DROP TABLE "tasks";
ALTER TABLE "new_tasks" RENAME TO "tasks";
CREATE INDEX "tasks_assignee_id_status_idx" ON "tasks"("assignee_id", "status");
CREATE INDEX "tasks_team_id_due_date_idx" ON "tasks"("team_id", "due_date");
CREATE INDEX "tasks_creator_id_idx" ON "tasks"("creator_id");
CREATE INDEX "tasks_status_priority_idx" ON "tasks"("status", "priority");
CREATE TABLE "new_teams" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "avatar" TEXT,
    "owner_id" TEXT NOT NULL,
    "settings" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "teams_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_teams" ("avatar", "created_at", "description", "id", "name", "owner_id", "updated_at") SELECT "avatar", "created_at", "description", "id", "name", "owner_id", "updated_at" FROM "teams";
DROP TABLE "teams";
ALTER TABLE "new_teams" RENAME TO "teams";
CREATE UNIQUE INDEX "teams_owner_id_key" ON "teams"("owner_id");
CREATE INDEX "teams_created_at_idx" ON "teams"("created_at");
CREATE INDEX "teams_owner_id_idx" ON "teams"("owner_id");
CREATE TABLE "new_user_achievements" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "user_id" TEXT NOT NULL,
    "achievement_id" TEXT NOT NULL,
    "unlocked_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "progress" TEXT,
    CONSTRAINT "user_achievements_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "user_achievements_achievement_id_fkey" FOREIGN KEY ("achievement_id") REFERENCES "achievements" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_user_achievements" ("achievement_id", "id", "progress", "unlocked_at", "user_id") SELECT "achievement_id", "id", "progress", "unlocked_at", "user_id" FROM "user_achievements";
DROP TABLE "user_achievements";
ALTER TABLE "new_user_achievements" RENAME TO "user_achievements";
CREATE INDEX "user_achievements_user_id_idx" ON "user_achievements"("user_id");
CREATE INDEX "user_achievements_achievement_id_idx" ON "user_achievements"("achievement_id");
CREATE UNIQUE INDEX "user_achievements_user_id_achievement_id_key" ON "user_achievements"("user_id", "achievement_id");
CREATE TABLE "new_users" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "email" TEXT NOT NULL,
    "password" TEXT,
    "firstName" TEXT NOT NULL,
    "lastName" TEXT NOT NULL,
    "avatar" TEXT,
    "teamId" TEXT,
    "level" INTEGER NOT NULL DEFAULT 1,
    "xp" INTEGER NOT NULL DEFAULT 0,
    "streak" INTEGER NOT NULL DEFAULT 0,
    "totalFocusTime" INTEGER NOT NULL DEFAULT 0,
    "tasksCompleted" INTEGER NOT NULL DEFAULT 0,
    "qualityScore" REAL NOT NULL DEFAULT 0,
    "wellnessScore" REAL NOT NULL DEFAULT 0,
    "preferences" TEXT,
    "settings" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL
);
INSERT INTO "new_users" ("avatar", "created_at", "email", "firstName", "id", "lastName", "level", "preferences", "qualityScore", "streak", "tasksCompleted", "totalFocusTime", "updated_at", "wellnessScore", "xp") SELECT "avatar", "created_at", "email", "firstName", "id", "lastName", "level", "preferences", "qualityScore", "streak", "tasksCompleted", "totalFocusTime", "updated_at", "wellnessScore", "xp" FROM "users";
DROP TABLE "users";
ALTER TABLE "new_users" RENAME TO "users";
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");
CREATE INDEX "users_email_idx" ON "users"("email");
CREATE INDEX "users_created_at_idx" ON "users"("created_at");
CREATE INDEX "users_level_xp_idx" ON "users"("level", "xp");
CREATE INDEX "users_teamId_idx" ON "users"("teamId");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

-- CreateIndex
CREATE UNIQUE INDEX "task_dependencies_dependent_task_id_prerequisite_id_key" ON "task_dependencies"("dependent_task_id", "prerequisite_id");

-- CreateIndex
CREATE INDEX "team_challenges_team_id_idx" ON "team_challenges"("team_id");

-- CreateIndex
CREATE INDEX "team_challenges_isActive_idx" ON "team_challenges"("isActive");

-- CreateIndex
CREATE INDEX "team_challenges_end_date_idx" ON "team_challenges"("end_date");

-- CreateIndex
CREATE INDEX "team_members_user_id_idx" ON "team_members"("user_id");
