# Missing Backend Endpoints Implementation Plan

## 🔍 Critical Discovery After Frontend Validation

### ✅ Current Working State
- **Sessions Module**: **COMPLETE** - All endpoints exist and work correctly
- **Analytics Module**: **COMPLETE** - All analytics endpoints work correctly
- **Auth & Users**: **COMPLETE** - Authentication and user management fully functional

### 🚨 Frontend Integration Issues Identified

The frontend DashboardScreen.tsx has **intentionally DISABLED** critical features:
```typescript
// Line 138-139: Sessions disabled
const sessions: any[] = []; // Mock empty sessions data
const sessionsLoading = false;

// Line 249: Achievements disabled
achievements={[]} // Disabled - using empty achievements array
```

### 📋 Validated Requirements from Frontend Code

## Phase 1: Re-enable Sessions Integration (URGENT - 1 Day)
**Issue**: Frontend uses mock sessions data but backend has complete sessions module

**Required Action**: Simply enable existing frontend API calls
```typescript
// Replace in DashboardScreen.tsx (lines 138-139):
const {
  data: sessions,
  isLoading: sessionsLoading
} = useGetSessionsQuery({ limit: 50 }); // Use existing API hook
```

**Expected Backend Endpoint**: Already exists at `/api/sessions`
- Sessions controller fully implemented ✅
- Session analytics endpoint already working ✅
- Database models complete ✅

## Phase 2: Achievements Module Implementation (HIGH PRIORITY - 2-3 Days)

### Frontend Requirements (from types/index.ts & apiSlice.ts):
```typescript
// Expected API endpoints:
GET /api/achievements
GET /api/users/{userId}/achievements
POST /api/achievements/{achievementId}/unlock

// Expected data structures:
interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: AchievementCategory;
  requirement: AchievementRequirement;
  xpReward: number;
  badgeUrl?: string;
}

interface UserAchievement {
  id: string;
  userId: string;
  achievementId: string;
  achievement: Achievement;
  unlockedAt: string;
  progress: number; // 0-100 percentage
}
```

### Database Schema Alignment (Already Exists):
```sql
-- Models already in schema.prisma:
Achievement {
  id, name, description, icon, category, xpValue, criteria, isActive
}
UserAchievement {
  id, userId, achievementId, unlockedAt, progress (JSON string)
}
```

### Implementation Files Required:
```
packages/backend/src/achievements/
├── achievements.module.ts
├── achievements.controller.ts
├── achievements.service.ts
├── dto/
│   ├── achievement.dto.ts
│   ├── user-achievement.dto.ts
│   └── index.ts
└── __tests__/
```

## Phase 3: Teams Module Implementation (HIGH PRIORITY - 2-3 Days)

### Frontend Requirements (from types/index.ts & apiSlice.ts):
```typescript
// Expected API endpoints:
GET /api/teams
GET /api/teams/{id}
POST /api/teams
POST /api/teams/{teamId}/join
POST /api/teams/{teamId}/leave

// Expected data structures:
interface Team {
  id: string;
  name: string;
  description?: string;
  avatar?: string;
  ownerId: string;
  members: TeamMember[];
  challenges: Challenge[];
}

interface TeamMember {
  id: string;
  userId: string;
  user: User;
  role: TeamRole; // 'OWNER' | 'ADMIN' | 'MEMBER'
  joinedAt: string;
}
```

### Database Schema Alignment (Already Exists):
```sql
-- Models already in schema.prisma:
Team {
  id, name, description, avatar, ownerId
}
TeamMember {
  id, userId, teamId, role, joinedAt
}
```

### Implementation Files Required:
```
packages/backend/src/teams/
├── teams.module.ts
├── teams.controller.ts
├── teams.service.ts
├── dto/
│   ├── team.dto.ts
│   ├── create-team.dto.ts
│   ├── join-team.dto.ts
│   └── index.ts
└── __tests__/
```

## Phase 4: Challenges Module Implementation (MEDIUM PRIORITY - 2-3 Days)

### Frontend Requirements (from types/index.ts & apiSlice.ts):
```typescript
// Expected API endpoints:
GET /api/challenges?teamId={teamId}&active={boolean}
GET /api/challenges/{id}
POST /api/challenges
POST /api/challenges/{challengeId}/join
POST /api/challenges/{challengeId}/progress

// Expected data structures:
interface Challenge {
  id: string;
  name: string;
  description: string;
  type: ChallengeType;
  targetValue: number;
  currentValue: number;
  startDate: string;
  endDate: string;
  participantIds: string[];
  participants: User[];
  rewards: ChallengeReward;
  createdBy: string;
}
```

### ⚠️ Database Schema Gap Identified:
**Current Schema**:
```sql
TeamChallenge {
  id, teamId, name, description, type, targetValue, currentValue
  startDate, endDate, isActive
}
```

**Required Additions**:
- Add `participantIds` field (JSON string for SQLite compatibility)
- Add `createdBy` field (references User.id)
- Add `rewards` field (JSON string for SQLite compatibility)

### Implementation Files Required:
```
packages/backend/src/challenges/
├── challenges.module.ts
├── challenges.controller.ts
├── challenges.service.ts
├── dto/
│   ├── challenge.dto.ts
│   ├── create-challenge.dto.ts
│   └── index.ts
└── __tests__/
```

## 🔧 Technical Implementation Details

### Module Structure Following Existing Patterns:
- Use existing `@ApiProperty` decorator patterns from auth DTOs
- Follow NestJS module structure (controller, service, DTOs)
- Implement `JwtAuthGuard` for authentication
- Use existing `DatabaseService` for Prisma operations
- Follow error handling patterns from sessions module

### Session Integration Fix:
```typescript
// DashboardScreen.tsx - Replace disabled sessions (lines 138-140):
import { useGetSessionsQuery } from '../../../store/api';

// Inside component:
const {
  data: sessions,
  isLoading: sessionsLoading,
  error: sessionsError
} = useGetSessionsQuery({
  filters: {
    type: ['POMODORO'],
    dateRange: dateRange
  },
  limit: 50
});
```

### Achievement Service Logic:
```typescript
// Achievement unlocking logic based on criteria
const checkAchievementProgress = async (userId: string, achievement: Achievement) => {
  const criteria = JSON.parse(achievement.criteria);

  switch (criteria.type) {
    case 'SESSION_COUNT':
      // Count user's sessions in timeframe
      break;
    case 'STREAK_DAYS':
      // Calculate current streak
      break;
    case 'TOTAL_TIME':
      // Sum focus time in minutes
      break;
  }
};
```

### Data Validation:
- ISO 8601 date string validation for challenges
- User authorization checks for team operations
- Progress percentage validation (0-100)
- Role-based access control for team operations

## 📊 Expected Outcomes

### Dashboard Integration Success:
- ✅ Sessions data populates weekly chart (currently empty)
- ✅ Achievement gallery displays real user achievements (currently empty)
- ✅ Team functionality works end-to-end
- ✅ Analytics includes gamification data

### API Completion:
- ✅ All frontend RTK Query endpoints implemented
- ✅ Consistent error handling and validation
- ✅ Complete OpenAPI documentation
- ✅ Production-ready authentication

### Frontend Code Re-activation:
```typescript
// DashboardScreen.tsx lines to fix:
achievements={userAchievements} // Replace empty array with real data
sessions={sessions}           // Replace empty array with real data
```

## ⚡ Implementation Timeline

### Day 1: Sessions Fix (URGENT)
- Re-enable frontend sessions API calls
- Test weekly chart population
- Verify session analytics integration

### Day 2-3: Achievements Module
- Create achievements controller/service/DTOs
- Implement achievement progress tracking
- Connect with user profile data

### Day 4-5: Teams Module
- Create teams controller/service/DTOs
- Implement team membership management
- Add team member validation

### Day 6-7: Challenges Module
- Database schema migration (add missing fields)
- Create challenges controller/service/DTOs
- Implement challenge progress tracking

### Day 8: Integration Testing
- End-to-end dashboard testing
- Performance optimization
- Error handling validation

## ✅ Success Criteria

1. **Frontend Dashboard**: All components populated with real data
2. **Sessions Integration**: Weekly chart shows actual session data
3. **Achievements System**: Users can unlock and view achievements
4. **Team Collaboration**: Team creation, joining, and management
5. **API Coverage**: 100% of frontend RTK Query endpoints implemented
6. **Test Coverage**: >90% for all new modules
7. **Documentation**: Complete OpenAPI/Swagger documentation

## 🎯 Immediate Actions Required

1. **Enable Sessions API** in frontend (1 hour fix)
2. **Create Achievements Module** (2-3 days)
3. **Create Teams Module** (2-3 days)
4. **Schema Migration** for challenges (1 day)
5. **Create Challenges Module** (2-3 days)
6. **Integration Testing** (1 day)

**Total Timeline**: 7-11 days

This plan is **validated against actual frontend code** and will enable all currently disabled dashboard features.