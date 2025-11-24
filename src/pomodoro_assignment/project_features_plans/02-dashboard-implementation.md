# Dashboard Screen Implementation Plan

## 🎯 Feature Overview

Implementation of the Progress Dashboard screen according to the provided UI/UX specifications, featuring real-time analytics integration and responsive Zen Garden theming.

### Priority: High 🔴
### Timeline: 5 days
### Dependencies: Backend Analytics Fixes (Issue #1-3)

---

## 📊 Dashboard Feature Specifications

### UI/UX Requirements (from ProductDesign.md & UIUX.md)

#### Screen Layouts:
- **Mobile (320px-768px)**: Single column, scrollable cards
- **Tablet (768px-1024px)**: Two-column layout with enhanced charts
- **Desktop (1024px+)**: Multi-column with full analytics

#### Core Components:
1. **Today's Focus**: Daily goal progress, streak counter, current session
2. **Wellness Metrics**: Hydration, movement, mood, stress, energy indicators
3. **Progress Charts**: Weekly focus trends, completion rates
4. **Achievement Gallery**: Recent badges and milestones
5. **Team Comparison**: Member rankings and collaboration metrics (future)

#### Design System:
- **Color Palette**: Zen Garden theme (Stone Gray, Moss Green, Water Blue)
- **Typography**: Inter font family with 8pt grid system
- **Components**: Atomic design with reusable elements
- **Animations**: Smooth transitions, zen-inspired progress indicators

---

## 🏗️ Technical Architecture

### Component Structure
```
packages/frontend/src/components/pages/DashboardScreen/
├── index.ts
├── DashboardScreen.tsx              // Main container component
├── hooks/
│   ├── useDashboardData.ts          // API integration logic
│   └── useDashboardLayout.ts        // Responsive layout logic
├── components/
│   ├── FocusAnalytics/
│   │   ├── FocusMetricsCard.tsx     // Today's focus metrics
│   │   ├── SessionStats.tsx         // Session statistics
│   │   └── GoalProgress.tsx         // Daily goal tracking
│   ├── Charts/
│   │   ├── WeeklyBarChart.tsx       // Weekly focus visualization
│   │   ├── ProgressRing.tsx         // Circular progress indicators
│   │   └── TrendIndicator.tsx       // Trend arrows and labels
│   ├── Wellness/
│   │   ├── WellnessCard.tsx         // Wellness metrics display
│   │   ├── MoodIndicator.tsx        // Mood/stress/energy levels
│   │   └── HydrationTracker.tsx     // Hydration progress
│   └── Achievements/
│       ├── AchievementGallery.tsx   // Recent achievements
│       ├── StreakCounter.tsx        // Streak visualization
│       └── LevelProgress.tsx        // XP and level display
└── utils/
    ├── dataFormatters.ts            // Time and percentage formatting
    └── chartHelpers.ts              // Chart data transformations
```

### API Integration Strategy
```typescript
// Real-time data hooks using existing RTK Query
import {
  useGetFocusAnalyticsQuery,
  useGetWellnessAnalyticsQuery,
  useGetTeamAnalyticsQuery
} from '../../../store/api/apiSlice';

// Custom hook for dashboard data aggregation
export const useDashboardData = (dateRange?: DateRange) => {
  const { data: focusData, isLoading: focusLoading, error: focusError } =
    useGetFocusAnalyticsQuery(dateRange);

  const { data: wellnessData, isLoading: wellnessLoading } =
    useGetWellnessAnalyticsQuery(dateRange);

  // Aggregate and format data for components
  return {
    focusMetrics: formatFocusMetrics(focusData),
    wellnessMetrics: formatWellnessMetrics(wellnessData),
    isLoading: focusLoading || wellnessLoading,
    error: focusError
  };
};
```

---

## 📱 Responsive Implementation

### Mobile Layout (320px - 768px)
Based on UI/UX ASCII design:

```
┌─────────────────────────────────────┐
│ 📊 Progress Dashboard               │
├─────────────────────────────────────┤
│ ┌─ Today's Focus ─────────────────┐ │
│ │ ⏱️ 3h 45m / 5h goal             │ │
│ │ ████████████░░░░ 75%            │ │
│ │ 🔥 5 day streak! 🎉             │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─ Wellness Metrics ───────────────┐ │
│ │ 🧘 Mindfulness: 15/30 min        │ │
│ │ 💧 Hydration: 6/8 glasses        │ │
│ │ 🚶 Movement: 800/1000 steps      │ │
│ │ 😊 Mood: Productive ⭐⭐⭐⭐        │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─ Recent Achievements ────────────┐ │
│ │ 🏆 Deep Focus Master             │ │
│ │ 🎯 Deadline Destroyer            │ │
│ │ 🌅 Early Bird                   │ │
│ │ 🧘 Zen Moment                    │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─ Weekly Overview ─────────────────┐ │
│ │ Mon Tue Wed Thu Fri Sat Sun       │ │
│ │ ✅ ✅ ✅ 📊 ⏳ 📅 📅             │ │
│ │ 25m 45m 60m 75m 90m 120m 60m     │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│  🏠  ⏰  📋  📊  👥  ⚙️             │
└─────────────────────────────────────┘
```

#### Implementation Details:
```typescript
// Mobile-First CSS Grid Layout
const DashboardGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: ${({ theme }) => theme.spacing.mobile.md};
  padding: ${({ theme }) => theme.spacing.mobile.sm};

  ${({ theme }) => theme.mediaQueries.tablet} {
    grid-template-columns: 1fr 1fr;
    gap: ${({ theme }) => theme.spacing.tablet.lg};
    padding: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    grid-template-columns: repeat(3, 1fr);
    gap: ${({ theme }) => theme.spacing.lg};
    padding: ${({ theme }) => theme.spacing.lg};
  }
`;
```

### Tablet Layout (768px - 1024px)
Two-column layout with enhanced visualizations:
- Left column: Focus metrics + weekly chart
- Right column: Wellness metrics + achievements

### Desktop Layout (1024px+)
Full multi-column dashboard with all features enabled.

---

## 🎨 Component Implementation Details

### 1. Focus Metrics Card
**File**: `FocusMetricsCard.tsx`

```typescript
interface FocusMetricsCardProps {
  dailyFocusTime: number;      // From API: minutes today
  weeklyFocusTime: number;     // From API: minutes this week
  monthlyFocusTime: number;    // From API: minutes this month
  averageSessionLength: number;// From API: minutes
  completionRate: number;      // From API: percentage
  focusTrend: TrendType;       // From API: IMPROVING|DECLINING|STABLE
  streak: number;              // From user profile
  dailyGoal: number;           // From user preferences (default 300)
}

const FocusMetricsCard: React.FC<FocusMetricsCardProps> = ({
  dailyFocusTime,
  weeklyFocusTime,
  monthlyFocusTime,
  averageSessionLength,
  completionRate,
  focusTrend,
  streak,
  dailyGoal = 300 // 5 hours default
}) => {
  const progressPercentage = Math.min(100, (dailyFocusTime / dailyGoal) * 100);

  return (
    <Card variant="primary">
      <CardHeader>
        <h3>Today's Focus</h3>
        <TrendIndicator trend={focusTrend} />
      </CardHeader>

      <ProgressRing
        progress={progressPercentage}
        size={120}
        strokeWidth={8}
        color={theme.colors.primary}
      >
        <TimeDisplay minutes={dailyFocusTime} />
        <GoalDisplay current={dailyFocusTime} goal={dailyGoal} />
      </ProgressRing>

      <StreakDisplay days={streak} />

      <StatsGrid>
        <StatItem label="Weekly Focus" value={formatMinutes(weeklyFocusTime)} />
        <StatItem label="Avg Session" value={`${averageSessionLength}m`} />
        <StatItem label="Completion" value={`${completionRate}%`} />
        <StatItem label="Monthly Total" value={formatMinutes(monthlyFocusTime)} />
      </StatsGrid>
    </Card>
  );
};
```

### 2. Weekly Bar Chart Component
**File**: `WeeklyBarChart.tsx`

```typescript
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface WeeklyBarChartProps {
  weeklyData: WeeklyData[];
  dailyGoal: number;
}

interface WeeklyData {
  day: string;           // 'Mon', 'Tue', etc.
  focusTime: number;     // Actual minutes
  goal: number;          // Daily goal in minutes
  completed: boolean;    // Met daily goal?
}

const WeeklyBarChart: React.FC<WeeklyBarChartProps> = ({ weeklyData, dailyGoal }) => {
  const transformedData = weeklyData.map(day => ({
    ...day,
    percentage: Math.min(100, (day.focusTime / day.goal) * 100)
  }));

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={transformedData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.border} />
        <XAxis
          dataKey="day"
          tick={{ fill: theme.colors.textSecondary, fontSize: 12 }}
        />
        <YAxis
          tick={{ fill: theme.colors.textSecondary, fontSize: 12 }}
          label={{ value: 'Minutes', angle: -90, position: 'insideLeft' }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: theme.colors.background,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borderRadius.md
          }}
        />
        <Bar
          dataKey="focusTime"
          fill={theme.colors.primary}
          radius={[8, 8, 0, 0]}
        />
        <Bar
          dataKey="goal"
          fill={theme.colors.secondary}
          opacity={0.3}
          radius={[8, 8, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
};
```

### 3. Wellness Metrics Card
**File**: `WellnessCard.tsx`

```typescript
interface WellnessCardProps {
  mindfulnessMinutes: number;
  hydrationGoal: number;
  hydrationCurrent: number;
  movementGoal: number;
  movementCurrent: number;
  moodRating: number;        // 1-5
  stressLevel: number;       // 1-5
  energyLevel: number;       // 1-5
}

const WellnessCard: React.FC<WellnessCardProps> = ({
  mindfulnessMinutes,
  hydrationGoal,
  hydrationCurrent,
  movementGoal,
  movementCurrent,
  moodRating,
  stressLevel,
  energyLevel
}) => {
  return (
    <Card variant="secondary">
      <CardHeader>
        <h3>Wellness Metrics</h3>
        <IconButton icon="⚙️" size="small" />
      </CardHeader>

      <WellnessGrid>
        <WellnessMetric
          icon="🧘"
          label="Mindfulness"
          current={mindfulnessMinutes}
          goal={30}
          unit="min"
          color={theme.colors.sageGreen}
        />

        <WellnessMetric
          icon="💧"
          label="Hydration"
          current={hydrationCurrent}
          goal={hydrationGoal}
          unit="glasses"
          color={theme.colors.waterBlue}
        />

        <WellnessMetric
          icon="🚶"
          label="Movement"
          current={movementCurrent}
          goal={movementGoal}
          unit="breaks"
          color={theme.colors.sunriseOrange}
        />
      </WellnessGrid>

      <MoodEnergyGrid>
        <MoodIndicator
          label="Mood"
          value={moodRating}
          icon="😊"
          color={theme.colors.primary}
        />

        <MoodIndicator
          label="Stress"
          value={6 - stressLevel} // Invert for better UX
          icon="😌"
          color={theme.colors.warmSand}
        />

        <MoodIndicator
          label="Energy"
          value={energyLevel}
          icon="⚡"
          color={theme.colors.sunriseOrange}
        />
      </MoodEnergyGrid>
    </Card>
  );
};
```

### 4. Achievement Gallery
**File**: `AchievementGallery.tsx`

```typescript
interface AchievementGalleryProps {
  achievements: UserAchievement[];
  level: number;
  xp: number;
  xpToNextLevel: number;
}

const AchievementGallery: React.FC<AchievementGalleryProps> = ({
  achievements,
  level,
  xp,
  xpToNextLevel
}) => {
  const recentAchievements = achievements
    .slice(-4)
    .reverse(); // Most recent first

  return (
    <Card variant="accent">
      <CardHeader>
        <h3>Recent Achievements</h3>
        <LevelDisplay level={level} />
      </CardHeader>

      <AchievementGrid>
        {recentAchievements.map((achievement) => (
          <AchievementBadge
            key={achievement.id}
            icon={achievement.achievement.icon}
            name={achievement.achievement.name}
            description={achievement.achievement.description}
            unlockedAt={achievement.unlockedAt}
          />
        ))}
      </AchievementGrid>

      <XPProgress
        current={xp}
        total={xpToNextLevel}
        level={level}
      />
    </Card>
  );
};
```

---

## 🎯 Data Integration

### API Response Handling
```typescript
// Expected API responses based on backend analysis:
interface FocusAnalyticsResponse {
  dailyFocusTime: number;      // 225 (minutes today)
  weeklyFocusTime: number;     // 1575 (minutes this week)
  monthlyFocusTime: number;    // 6750 (minutes this month)
  averageSessionLength: number; // 25.5 (minutes)
  peakFocusHours: number[];    // [9, 10, 14] (best hours)
  focusTrend: 'IMPROVING' | 'DECLINING' | 'STABLE';
  completionRate: number;      // 85.5 (percentage)
}

interface WellnessAnalyticsResponse {
  mindfulnessMinutes: number;  // 45 (minutes)
  hydrationGoal: number;       // 8 (glasses)
  hydrationCurrent: number;    // 6 (current intake)
  movementGoal: number;        // 5 (breaks)
  movementCurrent: number;     // 3 (breaks taken)
  moodRating: number;          // 4 (1-5 scale)
  stressLevel: number;         // 2 (1-5 scale)
  energyLevel: number;         // 4 (1-5 scale)
}
```

### Data Formatting Utilities
**File**: `utils/dataFormatters.ts`

```typescript
export const formatMinutes = (minutes: number): string => {
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
};

export const formatTimeOfDay = (minutes: number): string => {
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  const period = hours >= 12 ? 'PM' : 'AM';
  const displayHours = hours > 12 ? hours - 12 : hours || 12;
  return `${displayHours}:${mins.toString().padStart(2, '0')} ${period}`;
};

export const calculatePercentage = (current: number, goal: number): number => {
  return Math.min(100, Math.max(0, Math.round((current / goal) * 100)));
};

export const getTrendColor = (trend: 'IMPROVING' | 'DECLINING' | 'STABLE'): string => {
  switch (trend) {
    case 'IMPROVING': return theme.colors.success;
    case 'DECLINING': return theme.colors.error;
    case 'STABLE': return theme.colors.textSecondary;
    default: return theme.colors.textSecondary;
  }
};
```

---

## 🎨 Styling & Theming

### Zen Garden Design System Integration

```typescript
// theme extensions for dashboard components
export const dashboardTheme = {
  colors: {
    primary: '#7A8B7F',      // Moss Green
    secondary: '#6B8E9F',    // Water Blue
    accent: '#E67E50',        // Sunrise Orange
    success: '#7FA870',      // Sage Green
    warning: '#F4A261',      // Warm Amber
    error: '#C85A5A',        // Soft Red
    background: '#F4E4D4',    // Warm Sand
    surface: '#D4C5B9',      // Zen Stone
  },

  shadows: {
    card: '0 2px 8px rgba(0, 0, 0, 0.1)',
    elevated: '0 4px 16px rgba(0, 0, 0, 0.15)',
  },

  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px',
    xl: '16px',
  },

  spacing: {
    mobile: {
      xs: '4px',
      sm: '8px',
      md: '16px',
      lg: '24px',
      xl: '32px',
    },
    tablet: {
      xs: '6px',
      sm: '12px',
      md: '20px',
      lg: '32px',
      xl: '48px',
    },
    desktop: {
      xs: '8px',
      sm: '16px',
      md: '24px',
      lg: '32px',
      xl: '64px',
    },
  },
};
```

### Responsive Breakpoints
```typescript
export const mediaQueries = {
  mobile: '@media (max-width: 767px)',
  tablet: '@media (min-width: 768px) and (max-width: 1023px)',
  desktop: '@media (min-width: 1024px)',

  // Custom breakpoints for specific layouts
  mobileOnly: '@media (max-width: 425px)',
  tabletOnly: '@media (min-width: 426px) and (max-width: 1023px)',
};
```

---

## 🔄 State Management

### Redux Integration
```typescript
// Dashboard slice for local state management
interface DashboardState {
  selectedDateRange: DateRange;
  viewMode: 'overview' | 'detailed' | 'wellness';
  activeFilters: DashboardFilters;
  chartAnimations: boolean;
}

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    setDateRange: (state, action) => {
      state.selectedDateRange = action.payload;
    },
    setViewMode: (state, action) => {
      state.viewMode = action.payload;
    },
    toggleAnimations: (state) => {
      state.chartAnimations = !state.chartAnimations;
    },
  },
});
```

### Local Component State
```typescript
// For performance-critical components
const [hoveredMetric, setHoveredMetric] = useState<string | null>(null);
const [chartData, setChartData] = useState<ChartData[]>([]);
const [isTransitioning, setIsTransitioning] = useState(false);
```

---

## ⚡ Performance Optimization

### React Optimizations
```typescript
// Memoize expensive calculations
const memoizedChartData = useMemo(() => {
  return transformAnalyticsData(focusData);
}, [focusData]);

// Prevent unnecessary re-renders
const OptimizedChart = memo(({ data }: { data: ChartData[] }) => {
  return <WeeklyBarChart data={data} />;
});

// Lazy load heavy components
const AchievementGallery = lazy(() => import('./AchievementGallery'));
```

### Data Caching Strategy
```typescript
// RTK Query caching configuration
export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({ baseUrl: '/api' }),
  tagTypes: ['Analytics'],
  endpoints: (builder) => ({
    getFocusAnalytics: builder.query<FocusAnalytics, DateRange>({
      query: (dateRange) => ({
        url: 'analytics/focus',
        params: dateRange,
      }),
      // Cache for 5 minutes
      keepUnusedDataFor: 300,
      providesTags: ['Analytics'],
    }),
  }),
});
```

---

## 🧪 Testing Strategy

### Component Testing
```typescript
// FocusMetricsCard.test.tsx
describe('FocusMetricsCard', () => {
  it('should display daily focus time correctly', () => {
    const mockData = {
      dailyFocusTime: 225,
      weeklyFocusTime: 1575,
      // ... other props
    };

    render(<FocusMetricsCard {...mockData} />);

    expect(screen.getByText('3h 45m')).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
  });

  it('should show correct progress for daily goal', () => {
    const mockData = {
      dailyFocusTime: 150, // 2.5 hours
      dailyGoal: 300,      // 5 hours
      // ... other props
    };

    render(<FocusMetricsCard {...mockData} />);

    const progressRing = screen.getByRole('progressbar');
    expect(progressRing).toHaveAttribute('aria-valuenow', '50');
  });
});
```

### Integration Testing
```typescript
// DashboardScreen.integration.test.tsx
describe('DashboardScreen Integration', () => {
  it('should load and display analytics data', async () => {
    const mockFocusData = { /* mock focus analytics */ };
    const mockWellnessData = { /* mock wellness analytics */ };

    server.use(
      rest.get('/api/analytics/focus', (req, res, ctx) => {
        return res(ctx.json(mockFocusData));
      }),
      rest.get('/api/analytics/wellness', (req, res, ctx) => {
        return res(ctx.json(mockWellnessData));
      })
    );

    render(<DashboardScreen />);

    await waitFor(() => {
      expect(screen.getByText('Today\'s Focus')).toBeInTheDocument();
      expect(screen.getByText('Wellness Metrics')).toBeInTheDocument();
    });
  });
});
```

---

## 🚀 Implementation Timeline

### Day 1: Foundation Setup
- [ ] Create DashboardScreen component structure
- [ ] Set up responsive layout grid system
- [ ] Implement basic API integration hooks
- [ ] Add loading and error states

### Day 2: Focus Analytics Components
- [ ] Build FocusMetricsCard with real data integration
- [ ] Create custom ProgressRing component
- [ ] Implement SessionStats display
- [ ] Add Zen Garden styling

### Day 3: Data Visualization
- [ ] Install and configure Recharts
- [ ] Create WeeklyBarChart component
- [ ] Build TrendIndicator component
- [ ] Add interactive tooltips and animations

### Day 4: Wellness & Achievements
- [ ] Implement WellnessCard component
- [ ] Create AchievementGallery
- [ ] Add XP and level progression displays
- [ ] Integrate user preferences and goals

### Day 5: Polish & Testing
- [ ] Responsive design testing across breakpoints
- [ ] Performance optimization and memoization
- [ ] Accessibility compliance (ARIA labels, keyboard navigation)
- [ ] Error boundary implementation
- [ ] Comprehensive testing suite

---

## 📦 Dependencies

### New Packages Required
```bash
# Install charting library
pnpm add recharts

# Already available:
# - styled-components (theming)
# - RTK Query (API integration)
# - React Router (navigation)
# - TypeScript (type safety)
```

### Package.json Updates
```json
{
  "dependencies": {
    "recharts": "^2.8.0",
    // ... existing dependencies
  }
}
```

---

## ✅ Success Criteria

### Functional Requirements:
- ✅ Real-time data integration with analytics APIs
- ✅ Responsive design matching UI/UX ASCII layouts exactly
- ✅ Loading states and error handling for all data fetching
- ✅ Interactive data visualizations with Recharts
- ✅ Zen Garden theme consistency throughout

### Performance Requirements:
- ✅ Initial load time < 2 seconds
- ✅ Chart animations maintain 60fps
- ✅ Memory usage < 50MB for dashboard components
- ✅ API response handling < 500ms

### Accessibility Requirements:
- ✅ WCAG 2.1 AA compliance
- ✅ Keyboard navigation for all interactive elements
- ✅ Screen reader compatibility with proper ARIA labels
- ✅ High contrast mode support

### Code Quality:
- ✅ TypeScript strict mode compliance
- ✅ 90%+ test coverage
- ✅ ESLint + Prettier formatting
- ✅ Component documentation with Storybook stories

---

## 🔍 Debugging & Development

### Development Commands
```bash
# Start frontend development server
pnpm dev

# Run dashboard component tests
pnpm test -- --testPathPattern=DashboardScreen

# Build and test production build
pnpm build && pnpm preview

# Storybook for component development
pnpm storybook
```

### Debug Tools
- React DevTools for component inspection
- Redux DevTools for state management debugging
- Network tab for API request/response analysis
- Chrome DevTools Performance tab for optimization

---

**Last Updated**: 2025-01-24
**Status**: Ready for Implementation
**Dependencies**: Backend Analytics Fixes (Issues #1-3)
**Next Step**: Begin Day 1 foundation setup