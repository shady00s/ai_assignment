-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_sessions" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "user_id" TEXT NOT NULL,
    "task_id" TEXT,
    "type" TEXT NOT NULL DEFAULT 'POMODORO',
    "duration" INTEGER NOT NULL,
    "started_at" DATETIME,
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
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
