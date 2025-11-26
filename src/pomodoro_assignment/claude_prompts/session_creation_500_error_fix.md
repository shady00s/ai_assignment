# Session Creation 500 Error Fix

## Root Cause
The 500 Internal Server Error was caused by a database constraint violation. The `Session` model in the Prisma schema had `startTime` as a **required** field, but our backend service was modified to not automatically set `startTime` when creating sessions (to prevent auto-starting).

## Issue Details
- **Schema**: `startTime DateTime @map("started_at")` - Required field
- **Service**: Removed `startTime: new Date()` from `createSession()`
- **Result**: Database constraint violation when trying to create sessions without `startTime`

## Solution Applied

### 1. Schema Update
Modified `prisma/schema.prisma` to make `startTime` optional:
```prisma
// Before
startTime   DateTime    @map("started_at")   // Required

// After
startTime   DateTime?   @map("started_at")   // Optional, set when session starts
```

### 2. Database Migration
Created and applied migration `20251126070913_make_starttime_optional`:
```sql
-- SQLite migration
ALTER TABLE "sessions" ALTER COLUMN "started_at" DROP NOT NULL;
```

### 3. Service Updates
Updated `getActiveSession()` to only consider sessions that have actually started:
```typescript
where: {
  userId,
  completed: false,
  endTime: null,
  startTime: { not: null }, // Only consider started sessions
}
```

## Session Lifecycle Now

### Create Session
- Creates session without `startTime`
- Session exists but is not "active" yet
- No timer starts automatically

### Start Session
- Sets `startTime` to current timestamp
- Session becomes "active" in `getActiveSession()`
- Timer starts in frontend

### Skip/Complete Session
- Sets `endTime`
- Sets `completed` flag appropriately
- Session no longer appears in active queries

## Testing Verification
Session creation API call should now work:
```bash
POST http://localhost:3000/api/sessions
{
  "type": "SHORT_BREAK",
  "duration": 5
}
```

Expected response: 201 Created with session data (no 500 error)

## Benefits of This Approach
1. **Clear session lifecycle**: Create → Start → (Skip/Complete)
2. **No auto-starting**: Sessions only start when explicitly requested
3. **Proper state tracking**: Active sessions are truly active
4. **Flexible timer control**: Frontend controls when to actually start timing
5. **Better analytics**: Distinguishes between created vs started sessions