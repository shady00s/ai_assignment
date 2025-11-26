import React, { useMemo, useEffect } from 'react';
import styled from 'styled-components';
import {
  useGetFocusAnalyticsQuery,
  useGetWellnessAnalyticsQuery,
  useGetProfileQuery,
  useGetSessionsQuery,
  } from '@store/api';

import { useGetTodayWellnessQuery,
   } from '@store/api/wellnessApi';

import { LoadingSpinner } from '../../atoms/LoadingSpinner';
import { ErrorMessage } from '../../atoms/ErrorMessage';
import { FocusMetricsCard } from './components/FocusAnalytics/FocusMetricsCard';
import { WeeklyBarChart } from './components/Charts/WeeklyBarChart';
import { WellnessCard } from './components/Wellness/WellnessCard';
import { AchievementGallery } from './components/Achievements/AchievementGallery';
import { useCreateOrUpdateWellnessEntryMutation } from '@/store/api/apiSlice';

interface DateRange {
  startDate?: string;
  endDate?: string;
}

const DashboardContainer = styled.div`
  padding: ${({ theme }) => theme.spacing.mobile.sm};
  max-width: 100%;
  margin: 0 auto;
     min-height: 100vh;

  ${({ theme }) => theme.mediaQueries.tablet} {
    padding: ${({ theme }) => theme.spacing.tablet.md};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    padding: ${({ theme }) => theme.spacing.desktop.lg};
  }
`;

const DashboardHeader = styled.header`
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  text-align: center;

  h1 {
    color: ${({ theme }) => theme.colors.neutral[500]};
    font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
    margin-bottom: ${({ theme }) => theme.spacing.sm};

    ${({ theme }) => theme.mediaQueries.tablet} {
      font-size: ${({ theme }) => theme.typography.fontSize.tablet['3xl']};
    }

    ${({ theme }) => theme.mediaQueries.desktop} {
      font-size: ${({ theme }) => theme.typography.fontSize.desktop['3xl']};
    }
  }

  p {
    color: ${({ theme }) => theme.colors.neutral[400]};
    font-size: ${({ theme }) => theme.typography.fontSize.base};

    ${({ theme }) => theme.mediaQueries.tablet} {
      font-size: ${({ theme }) => theme.typography.fontSize.tablet.lg};
    }
  }
`;

const DashboardGrid = styled.div`
  display: grid;
    grid-template-columns: 1fr;
   gap: ${({ theme }) => theme.spacing.mobile.md};

  ${({ theme }) => theme.mediaQueries.tablet} {
    grid-template-columns: repeat(1, 1fr);
    gap: ${({ theme }) => theme.spacing.tablet.lg};
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    grid-template-columns: repeat(2, 1fr);
    gap: ${({ theme }) => theme.spacing.lg};
  }
`;

const FullWidthSection = styled.div`
  grid-column: 1 / -1;
`;

const WeeklyChartSection = styled.div`
  grid-column: 1 / -1;

  ${({ theme }) => theme.mediaQueries.tablet} {
    grid-column: 1 / span 2;
  }

  ${({ theme }) => theme.mediaQueries.desktop} {
    grid-column: 1 / span 2;
  }
`;

const WellnessSection = styled.div`
  ${({ theme }) => theme.mediaQueries.desktop} {
    grid-column: span 1;
  }
`;



export const DashboardScreen: React.FC = () => {
  const dateRange: DateRange = useMemo(() => {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 7); // Last 7 days

    return {
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString()
    };
  }, []);

  const {
    data: focusData,
    isLoading: focusLoading,
    error: focusError
  } = useGetFocusAnalyticsQuery(dateRange);

  const {
    data: wellnessData,
    isLoading: wellnessLoading,
    error: wellnessError
  } = useGetWellnessAnalyticsQuery(dateRange);

  const {
    data: todayWellness,
    isLoading: todayWellnessLoading,
    error: todayWellnessError
  } = useGetTodayWellnessQuery();

  const [createWellnessEntry, { isLoading: isCreatingWellness }] = useCreateOrUpdateWellnessEntryMutation();

  const {
    data: userProfile,
    isLoading: profileLoading,
    error: profileError
  } = useGetProfileQuery();

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

  // Create default wellness entry if none exists
  useEffect(() => {
    if (!todayWellnessLoading && !todayWellnessError && !todayWellness && !isCreatingWellness) {
      createWellnessEntry({
        hydrationGlasses: 0,
        hydrationGoal: 8,
        movementBreaks: 0,
        movementMinutes: 0,
        moodRating: 3,
        stressLevel: 3,
        energyLevel: 3,
        meditationMinutes: 0,
        breathingExercises: 0,
        mindfulnessSessions: 0,
        postureChecks: 0,
        eyeRestBreaks: 0,
      });
    }
  }, [todayWellness, todayWellnessLoading, todayWellnessError, isCreatingWellness, createWellnessEntry]);

  // Disable achievements API call - using empty data
  const userAchievements: any[] = [];

  // Use real sessions data from API
  const sessions = sessionsData || [];

  const isLoading = focusLoading || wellnessLoading || profileLoading || todayWellnessLoading || isCreatingWellness || sessionsLoading;
  const hasError = focusError || wellnessError || profileError || todayWellnessError || sessionsError;

  if (isLoading) {
    return (
      <DashboardContainer>
        <LoadingSpinner size="large" centered />
      </DashboardContainer>
    );
  }

  if (hasError) {
    return (
      <DashboardContainer>
        <ErrorMessage
          message="Failed to load dashboard data. Please try again later."
          variant="card"
        />
      </DashboardContainer>
    );
  }

  // Generate real weekly data from sessions
  const generateRealWeeklyData = () => {
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const today = new Date();
    const currentDayIndex = today.getDay() === 0 ? 6 : today.getDay() - 1; // Monday = 0

    return days.map((day, index) => {
      // Calculate the date for this day of the week
      const dayDate = new Date(today);
      dayDate.setDate(today.getDate() - currentDayIndex + index);
      const dayStart = new Date(dayDate);
      dayStart.setHours(0, 0, 0, 0);
      const dayEnd = new Date(dayDate);
      dayEnd.setHours(23, 59, 59, 999);

      // Filter sessions for this day from the fetched sessions
      const daySessions = sessions?.filter(session => {
        if (session.type !== 'POMODORO') return false;
        const sessionDate = new Date(session.startTime);
        return sessionDate >= dayStart && sessionDate <= dayEnd;
      }) || [];

      const focusTime = daySessions.reduce((total, session) => total + session.duration, 0);
      const dailyGoal = userProfile?.preferences?.workDuration ? userProfile.preferences.workDuration * 60 : 300; // Convert hours to minutes

      return {
        day,
        focusTime,
        goal: dailyGoal,
        completed: focusTime >= dailyGoal
      };
    });
  };

  const weeklyData = generateRealWeeklyData();
  const dailyGoal = userProfile?.preferences?.workDuration ? userProfile.preferences.workDuration * 60 : 300; // Convert hours to minutes

  return (
    <DashboardContainer>
      <DashboardHeader>
        <h1>Progress Dashboard</h1>
        <p>Track your focus, wellness, and achievements</p>
      </DashboardHeader>

      <DashboardGrid>
        {/* Today's Focus Card */}
        <FocusMetricsCard
          dailyFocusTime={focusData?.dailyFocusTime || 0}
          weeklyFocusTime={focusData?.weeklyFocusTime || 0}
          monthlyFocusTime={focusData?.monthlyFocusTime || 0}
          averageSessionLength={focusData?.averageSessionLength || 0}
          completionRate={focusData?.completionRate || 0}
          focusTrend={focusData?.focusTrend || 'STABLE'}
          streak={userProfile?.streak || 0}
          dailyGoal={dailyGoal}
        />
          {/* Wellness Metrics */}
        <WellnessSection>
          <WellnessCard
            mindfulnessMinutes={wellnessData?.mindfulnessMinutes || 0}
            hydrationGoal={wellnessData?.hydrationGoal || 8}
            hydrationCurrent={wellnessData?.hydrationCurrent || 0}
            movementGoal={wellnessData?.movementGoal || 5}
            movementCurrent={wellnessData?.movementCurrent || 0}
            moodRating={wellnessData?.moodRating || 3}
            stressLevel={wellnessData?.stressLevel || 3}
            energyLevel={wellnessData?.energyLevel || 3}
          />
        </WellnessSection>

        {/* Weekly Chart */}
        <WeeklyChartSection>
          <WeeklyBarChart
            weeklyData={weeklyData}
            dailyGoal={dailyGoal}
          />
        </WeeklyChartSection>

      

        {/* Achievement Gallery */}
        <FullWidthSection>
          <AchievementGallery
            level={userProfile?.level || 1}
            xp={userProfile?.xp || 0}
            achievements={userAchievements || []}
          />
        </FullWidthSection>
      </DashboardGrid>
    </DashboardContainer>
  );
};