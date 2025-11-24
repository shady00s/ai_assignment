-- CreateTable
CREATE TABLE "wellness_entries" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "user_id" TEXT NOT NULL,
    "date" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "hydrationGlasses" INTEGER NOT NULL DEFAULT 0,
    "hydrationGoal" INTEGER NOT NULL DEFAULT 8,
    "movementBreaks" INTEGER NOT NULL DEFAULT 0,
    "movementMinutes" INTEGER NOT NULL DEFAULT 0,
    "stepsCount" INTEGER,
    "meditationMinutes" INTEGER NOT NULL DEFAULT 0,
    "breathingExercises" INTEGER NOT NULL DEFAULT 0,
    "mindfulnessSessions" INTEGER NOT NULL DEFAULT 0,
    "moodRating" INTEGER NOT NULL DEFAULT 3,
    "stressLevel" INTEGER NOT NULL DEFAULT 3,
    "energyLevel" INTEGER NOT NULL DEFAULT 3,
    "sleepQuality" INTEGER,
    "sleepHours" REAL,
    "postureChecks" INTEGER NOT NULL DEFAULT 0,
    "eyeRestBreaks" INTEGER NOT NULL DEFAULT 0,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "wellness_entries_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "wellness_reminders" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "user_id" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "frequency" INTEGER NOT NULL,
    "startTime" TEXT NOT NULL,
    "endTime" TEXT NOT NULL,
    "weekdays" TEXT NOT NULL,
    "last_trigger" DATETIME,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "wellness_reminders_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "wellness_goals" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "user_id" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "targetValue" INTEGER NOT NULL,
    "period" TEXT NOT NULL,
    "active" BOOLEAN NOT NULL DEFAULT true,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "wellness_goals_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateIndex
CREATE INDEX "wellness_entries_user_id_date_idx" ON "wellness_entries"("user_id", "date");

-- CreateIndex
CREATE INDEX "wellness_entries_date_idx" ON "wellness_entries"("date");

-- CreateIndex
CREATE UNIQUE INDEX "wellness_entries_user_id_date_key" ON "wellness_entries"("user_id", "date");

-- CreateIndex
CREATE INDEX "wellness_reminders_user_id_idx" ON "wellness_reminders"("user_id");

-- CreateIndex
CREATE INDEX "wellness_reminders_type_idx" ON "wellness_reminders"("type");

-- CreateIndex
CREATE INDEX "wellness_goals_user_id_idx" ON "wellness_goals"("user_id");

-- CreateIndex
CREATE UNIQUE INDEX "wellness_goals_user_id_category_period_key" ON "wellness_goals"("user_id", "category", "period");
