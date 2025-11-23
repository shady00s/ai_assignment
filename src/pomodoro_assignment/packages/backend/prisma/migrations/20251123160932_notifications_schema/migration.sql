-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_notifications" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "user_id" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "read" BOOLEAN NOT NULL DEFAULT false,
    "entity_id" TEXT,
    "entity_type" TEXT,
    "data" TEXT,
    "read_at" DATETIME,
    "scheduled_for" DATETIME,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "notifications_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_notifications" ("created_at", "data", "id", "message", "read_at", "title", "type", "user_id") SELECT "created_at", "data", "id", "message", "read_at", "title", "type", "user_id" FROM "notifications";
DROP TABLE "notifications";
ALTER TABLE "new_notifications" RENAME TO "notifications";
CREATE INDEX "notifications_user_id_idx" ON "notifications"("user_id");
CREATE INDEX "notifications_read_idx" ON "notifications"("read");
CREATE INDEX "notifications_read_at_idx" ON "notifications"("read_at");
CREATE INDEX "notifications_scheduled_for_idx" ON "notifications"("scheduled_for");
CREATE INDEX "notifications_created_at_idx" ON "notifications"("created_at");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
