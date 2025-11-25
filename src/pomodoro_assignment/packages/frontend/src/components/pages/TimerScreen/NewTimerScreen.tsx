import React, { useState, useEffect } from 'react';
import styled, { createGlobalStyle } from 'styled-components';
import {
  MobileLayout,
  TabletLayout,
  DesktopLayout,
} from './components/ResponsiveLayout';
import {
  NotificationCenter,
} from './components/Header';
import {
  ZenGardenTimer,
  ZenElements,
} from './components/ZenGarden';
import {
  TaskDisplay,
  TaskProgress,
  EnergyIndicator,
} from './components/CurrentTask';
import {
  WellnessMetrics,
  HydrationTracker,
  MoodTracker,
  MovementTracker,
  MeditationTimer,
} from './components/WellnessPanel';
import {
  EnhancedControls,
  AmbientSettings,
  FocusMode,
} from './components/SessionControls';
import {
  TodayProgress,
  AchievementDisplay,
  StreakTracker,
} from './components/AnalyticsPanel';
import {
  useTimerLogic,
  useWellnessData,
  useTaskIntegration,
  useResponsiveLayout,
} from './hooks';
import { useGetProfileQuery } from '@/store/api/apiSlice';
 
// Global styles for animations
const GlobalStyles = createGlobalStyle`
  @keyframes pulse {
    0%, 100% {
      opacity: 0;
      transform: scale(0.95);
    }
    50% {
      opacity: 1;
      transform: scale(1.05);
    }
  }

  @keyframes shimmer {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(100%);
    }
  }

  @keyframes waterDrop {
    0%, 100% {
      transform: translateY(0) scale(1);
    }
    50% {
      transform: translateY(-2px) scale(1.1);
    }
  }

  @keyframes float {
    0%, 100% {
      transform: translateY(0px) translateX(0px) rotate(0deg);
    }
    25% {
      transform: translateY(-10px) translateX(5px) rotate(90deg);
    }
    50% {
      transform: translateY(-5px) translateX(-5px) rotate(180deg);
    }
    75% {
      transform: translateY(-15px) translateX(3px) rotate(270deg);
    }
  }

  @keyframes ripple {
    0% {
      transform: scale(0.8);
      opacity: 0;
    }
    50% {
      opacity: 0.3;
    }
    100% {
      transform: scale(2);
      opacity: 0;
    }
  }
`;

const TimerScreenContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #F8F9FA 0%, #E8D8C8 100%);
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
  overflow-x: hidden;
`;

const LoadingOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  font-size: 18px;
  color: #2C3E50;
`;

interface NewTimerScreenProps {
  className?: string;
}

// Mock data for demonstration - replace with real API calls
const mockAchievements = [
  {
    id: '1',
    name: 'Deep Focus Master',
    description: 'Completed a 2-hour focus session',
    icon: '🏆',
    category: 'FOCUS' as const,
    unlockedAt: new Date().toISOString(),
    progress: 100,
    isNew: true,
  },
  {
    id: '2',
    name: 'Early Bird',
    description: 'Started 3 sessions before 9 AM',
    icon: '🌅',
    category: 'CONSISTENCY' as const,
    unlockedAt: new Date(Date.now() - 86400000).toISOString(),
    progress: 100,
    isNew: false,
  },
];

const mockNotifications = [
  {
    id: '1',
    title: 'Achievement Unlocked!',
    message: 'You completed your first 2-hour focus session',
    type: 'achievement' as const,
    timestamp: new Date().toISOString(),
    read: false,
  },
  {
    id: '2',
    title: 'Team Update',
    message: 'Sarah completed the API integration task',
    type: 'team_update' as const,
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    read: false,
  },
];

export const NewTimerScreen: React.FC<NewTimerScreenProps> = ({ className }) => {
  // Layout hooks
  const { breakpoint, isMobile, isTablet, isDesktop } = useResponsiveLayout();

  // Timer logic
  const timerLogic = useTimerLogic({
    onSessionComplete: (sessionId) => {
      console.log('Session completed:', sessionId);
    },
  });

  // Task integration
  const taskIntegration = useTaskIntegration({
    priority: 'HIGH',
    autoRefresh: true,
  });

  // Wellness data
  const wellnessData = useWellnessData({
    autoRefresh: true,
  });

  // User profile
  const { data: userProfile } = useGetProfileQuery();

  // Local state
  const [showNotifications, setShowNotifications] = useState(false);
  const [focusModeActive, setFocusModeActive] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [volume, setVolume] = useState(70);
  const [ambientSound, setAmbientSound] = useState('forest');

  // Mock analytics data - replace with real API calls
  const analyticsData = {
    focusTimeMinutes: 225, // 3h 45m
    focusTimeGoal: 300, // 5h
    tasksCompleted: 4,
    tasksTotal: 6,
    streakDays: 5,
    weeklyTrend: 'up' as const,
    qualityScore: 92,
  };

  const streakData = {
    currentStreak: 5,
    longestStreak: 23,
    streakHistory: Array.from({ length: 30 }, (_, i) => i < 25), // Last 25 days completed
    todayCompleted: false,
  };

  
  // Zen Garden timer with ambient elements
  const ZenGardenComponent = (
    <div style={{ position: 'relative' }}>
      <ZenElements
        isRunning={timerLogic.isRunning}
        sessionType={timerLogic.sessionType}
        progress={timerLogic.progress}
      />
      <ZenGardenTimer
        remainingTime={timerLogic.remainingTime}
        totalTime={timerLogic.totalTime}
        sessionType={timerLogic.sessionType}
        isRunning={timerLogic.isRunning}
        isPaused={timerLogic.isPaused}
        progress={timerLogic.progress}
        sessionsCompleted={timerLogic.sessionsCompleted}
      />
    </div>
  );

  // Current task display
  const CurrentTaskComponent = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <TaskDisplay
        currentTask={taskIntegration.currentTask}
        sessionCount={timerLogic.sessionsCompleted}
        totalSessions={4}
        energyLevel="HIGH"
        isTimerRunning={timerLogic.isRunning}
        onTaskSelect={() => {
          const nextTask = taskIntegration.getNextTask();
          if (nextTask) {
            taskIntegration.selectTask(nextTask.id);
          }
        }}
      />
      {taskIntegration.currentTask && (
        <TaskProgress
          task={taskIntegration.currentTask}
          sessionCount={timerLogic.sessionsCompleted}
          totalSessions={4}
          showTimeEstimate={!isMobile}
        />
      )}
    </div>
  );

  // Enhanced session controls
  const SessionControlsComponent = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <EnhancedControls
        isRunning={timerLogic.isRunning}
        isPaused={timerLogic.isPaused}
        canStart={timerLogic.canStart}
        canPause={timerLogic.canPause}
        canSkip={timerLogic.canSkip}
        canComplete={timerLogic.canComplete}
        sessionType={timerLogic.sessionType}
        onStart={timerLogic.handleStart}
        onPause={timerLogic.handlePause}
        onResume={timerLogic.handleResume}
        onSkip={timerLogic.handleSkip}
        onComplete={timerLogic.handleComplete}
      />
      {isDesktop && (
        <AmbientSettings
          soundEnabled={soundEnabled}
          volume={volume}
          currentSound={ambientSound}
          focusMode={focusModeActive}
          notificationsEnabled={true}
          onSoundToggle={setSoundEnabled}
          onVolumeChange={setVolume}
          onSoundChange={setAmbientSound}
          onFocusModeToggle={setFocusModeActive}
          onNotificationsToggle={(enabled) => console.log('Notifications:', enabled)}
        />
      )}
    </div>
  );

  // Analytics components
  const AnalyticsComponent = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <TodayProgress {...analyticsData} />
      {isTablet && <AchievementDisplay achievements={mockAchievements} />}
      {isDesktop && <StreakTracker {...streakData} />}
    </div>
  );

  // Wellness components
  const WellnessComponent = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {wellnessData.wellnessData && (
        <>
          <WellnessMetrics
            wellnessData={wellnessData.wellnessData}
            onIncrementHydration={() => wellnessData.incrementHydration(1)}
            onStartMovement={() => wellnessData.logMovement(5, 'stretch', 'LOW')}
            onUpdateMood={(mood) => wellnessData.updateMood(mood)}
            onStartMeditation={() => wellnessData.logMeditation(5, 'breathing', 4)}
            compact={isMobile}
          />
          {isTablet && (
            <HydrationTracker
              currentGlasses={wellnessData.wellnessData.hydrationGlasses}
              dailyGoal={wellnessData.wellnessData.hydrationGoal}
              glassSize={250}
              onIncrement={(glasses) => wellnessData.incrementHydration(glasses - wellnessData.wellnessData.hydrationGlasses)}
              compact={false}
            />
          )}
          {isDesktop && (
            <>
              <MoodTracker
                moodRating={wellnessData.wellnessData.moodRating}
                stressLevel={wellnessData.wellnessData.stressLevel}
                energyLevel={wellnessData.wellnessData.energyLevel}
                onUpdateMood={(mood) => wellnessData.updateMood(mood)}
                compact={false}
              />
              <FocusMode
                isActive={focusModeActive}
                onToggle={setFocusModeActive}
                blockNotifications={true}
                enableAmbientSounds={soundEnabled}
                setAmbientSound={setAmbientSound}
              />
            </>
          )}
        </>
      )}
    </div>
  );

  if (wellnessData.isLoading || taskIntegration.isLoading) {
    return (
      <TimerScreenContainer className={className}>
        <GlobalStyles />
        <LoadingOverlay>Loading Timer...</LoadingOverlay>
      </TimerScreenContainer>
    );
  }

  // Render responsive layout
  const renderLayout = () => {
    switch (breakpoint) {
      case 'mobile':
        return (
          <MobileLayout
            zenGarden={ZenGardenComponent}
            currentTask={CurrentTaskComponent}
            sessionControls={SessionControlsComponent}
            analytics={AnalyticsComponent}
            wellness={WellnessComponent}
          />
        );
      case 'tablet':
        return (
          <TabletLayout
            zenGarden={ZenGardenComponent}
            currentTask={CurrentTaskComponent}
            sessionControls={SessionControlsComponent}
            analytics={AnalyticsComponent}
            wellness={WellnessComponent}
          />
        );
      case 'desktop':
      default:
        return (
          <DesktopLayout
            zenGarden={ZenGardenComponent}
            currentTask={CurrentTaskComponent}
            sessionControls={SessionControlsComponent}
            analytics={AnalyticsComponent}
            wellness={WellnessComponent}
          />
        );
    }
  };

  return (
    <TimerScreenContainer className={className}>
      <GlobalStyles />
      {renderLayout()}

      {showNotifications && (
        <div style={{
          position: 'fixed',
          top: '60px',
          right: '20px',
          zIndex: 1000,
        }}>
          <NotificationCenter
            notifications={mockNotifications}
            onClose={() => setShowNotifications(false)}
          />
        </div>
      )}
    </TimerScreenContainer>
  );
};

export type { NewTimerScreenProps };