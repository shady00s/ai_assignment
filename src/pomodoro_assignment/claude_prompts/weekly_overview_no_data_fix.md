# Fix Weekly Overview No Data Issue

## User Request
The user reported that the weekly overview chart shows no data, despite having the application functional in other areas.

## Root Cause Analysis
The weekly overview chart was displaying no data because the sessions API call was temporarily disabled in the DashboardScreen component:

**Problem Location**: `DashboardScreen.tsx:171`
```tsx
// Temporarily disable sessions API call - using empty data
const sessions: any[] = []; // Mock empty sessions data
```

**Impact Chain**:
1. `generateRealWeeklyData()` function depends on sessions data to calculate weekly focus time
2. Function filters sessions by `type: 'POMODORO'` and date ranges
3. With empty sessions array, all `focusTime` values were calculated as 0
4. Weekly bar chart displayed empty bars with no data

## Backend API Verification
- **Sessions Controller**: Available at `GET /api/sessions` with query parameters
- **Query Parameters**: `type`, `startDate`, `endDate` for filtering
- **Database Schema**: Sessions include `id`, `userId`, `type`, `duration`, `startTime`, `endTime`, `completed`
- **Frontend API**: `useGetSessionsQuery` available in `@store/api`

## Solution Implemented

### 1. Added Sessions API Import
```tsx
import {
  useGetFocusAnalyticsQuery,
  useGetWellnessAnalyticsQuery,
  useGetProfileQuery,
  useGetSessionsQuery,  // Added this
} from '@store/api';
```

### 2. Implemented Sessions API Call
```tsx
// Get sessions data for weekly chart
const {
  data: sessionsData,
  isLoading: sessionsLoading,
  error: sessionsError
} = useGetSessionsQuery({
  type: 'POMODORO',
  startDate: dateRange.startDate,
  endDate: dateRange.endDate
});
```

### 3. Replaced Mock Data with Real Data
```tsx
// Before: const sessions: any[] = []; // Mock empty sessions data
// After:
const sessions = sessionsData || [];
```

### 4. Updated Loading and Error States
```tsx
const isLoading = focusLoading || wellnessLoading || profileLoading || todayWellnessLoading || isCreatingWellness || sessionsLoading;
const hasError = focusError || wellnessError || profileError || todayWellnessError || sessionsError;
```

## Expected Data Flow
1. **API Request**: `GET /api/sessions?type=POMODORO&startDate=...&endDate=...`
2. **Response**: Array of session objects with focus session data
3. **Processing**: `generateRealWeeklyData()` filters and processes by day
4. **Display**: Weekly bar chart shows focus time for each day

## Files Modified
- `/packages/frontend/src/components/pages/DashboardScreen/DashboardScreen.tsx`

## Testing Scenarios
1. **No Sessions**: Chart should show empty bars (same as before but intentional)
2. **Some Sessions**: Chart should show progress bars with appropriate colors
3. **API Error**: Error state should be handled with error message
4. **API Loading**: Loading spinner should show during data fetch

## Color Coding Logic
The chart uses color coding based on goal completion:
- **Green** (100%+): Goal achieved or exceeded
- **Moss Green** (75-99%): Good progress
- **Amber** (50-74%): Needs improvement
- **Red** (0-49%): Significant gap

## Result
- Weekly overview now displays actual focus session data
- Loading states properly handle sessions API calls
- Error handling covers sessions API failures
- Maintains existing functionality while adding real data