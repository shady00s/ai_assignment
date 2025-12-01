import React, { useState, useEffect, useMemo } from 'react';
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
  useAmbientSound,
} from './hooks';
import { TaskSelectionModal } from '@/components/organisms/TaskSelectionModal';
import {
  useGetProfileQuery,
  useGetFocusAnalyticsQuery,
  useGetTaskAnalyticsQuery,
  useGetSessionAnalyticsQuery,
  useGetWellnessHistoryQuery,
  useGetWellnessAnalyticsQuery,
  useGetWellnessRecommendationsQuery,
  useUpdateTaskMutation,
} from '@/store/api/apiSlice';
 
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

  /* Dark mode styles */
  .dark-mode & {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
  }
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

  /* Dark mode styles */
  .dark-mode & {
    background: rgba(30, 41, 59, 0.9) !important;
  }
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
 // Task integration
  const taskIntegration = useTaskIntegration({
    autoRefresh: true,
  });
  // Timer logic
  const timerLogic = useTimerLogic({
    currentTaskId: taskIntegration.currentTask?.id,
    onSessionComplete: async (sessionId) => {
      console.log('Session completed:', sessionId);

      // When a POMODORO session completes, increment the task's completed pomodoros
      if (taskIntegration.currentTask && timerLogic.sessionType === 'POMODORO') {
        try {
          const newCompletedPomodoros = taskIntegration.currentTask.completedPomodoros + 1;

          // Check if task is now complete
          const isTaskComplete = newCompletedPomodoros >= taskIntegration.currentTask.estimatedPomodoros;

          await updateTask({
            id: taskIntegration.currentTask.id,
            updates: {
              completedPomodoros: newCompletedPomodoros,
              status: isTaskComplete ? 'COMPLETED' : 'IN_PROGRESS'
            }
          }).unwrap();

          console.log(`Task ${taskIntegration.currentTask.id} updated: ${newCompletedPomodoros}/${taskIntegration.currentTask.estimatedPomodoros} pomodoros completed`);

          // If task is complete, automatically select the next available task
          if (isTaskComplete) {
            setTimeout(() => {
              const nextTask = taskIntegration.getNextTask();
              if (nextTask) {
                taskIntegration.selectTask(nextTask.id);
                console.log('Automatically selected next task:', nextTask.title);
              }
            }, 2000); // Wait 2 seconds before selecting next task
          }

        } catch (error) {
          console.error('Failed to update task after session completion:', error);
        }
      }
    },
  });

 

  // Wellness data
  const wellnessData = useWellnessData({
    autoRefresh: true,
  });

  // User profile
  const { data: userProfile } = useGetProfileQuery();

  // Analytics data from API
  const { data: focusAnalytics, isLoading: focusAnalyticsLoading, error: focusAnalyticsError } = useGetFocusAnalyticsQuery({});
  const { data: taskAnalytics, isLoading: taskAnalyticsLoading, error: taskAnalyticsError } = useGetTaskAnalyticsQuery({});
  const { data: sessionAnalytics, isLoading: sessionAnalyticsLoading, error: sessionAnalyticsError } = useGetSessionAnalyticsQuery({});

  // Memoize date calculations for wellness queries
  const wellnessQueryParams = useMemo(() => ({
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    endDate: new Date().toISOString().split('T')[0]
  }), []);

  // Wellness data from API (same as DashboardScreen)
  const { data: wellnessAnalytics, isLoading: wellnessAnalyticsLoading, error: wellnessAnalyticsError } = useGetWellnessAnalyticsQuery(wellnessQueryParams);

  const { data: wellnessHistory } = useGetWellnessHistoryQuery({
    days: 30
  });

  const { data: wellnessRecommendations } = useGetWellnessRecommendationsQuery({
    days: 30
  });

  // Task update mutation for pomodoro completion
  const [updateTask] = useUpdateTaskMutation();

  // Local state - initialize with user preferences if available
  const [showNotifications, setShowNotifications] = useState(false);
  const [focusModeActive, setFocusModeActive] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(userProfile?.preferences?.soundEnabled || false);
  const [volume, setVolume] = useState(userProfile?.preferences?.volume || 70);
  const [ambientSoundType, setAmbientSoundType] = useState(userProfile?.preferences?.ambientSound || 'forest');
  const [showTaskSelectionModal, setShowTaskSelectionModal] = useState(false);

  // Process analytics data from API responses
  const analyticsData = {
    focusTimeMinutes: focusAnalytics?.dailyFocusTime || 0,
    focusTimeGoal: 300, // 5h default goal - could be made configurable
    tasksCompleted: taskAnalytics?.completedTasks || 0,
    tasksTotal: taskAnalytics?.totalTasks || 0,
    streakDays: userProfile?.streak || 0,
    weeklyTrend: focusAnalytics?.focusTrend === 'IMPROVING' ? 'up' as const :
                  focusAnalytics?.focusTrend === 'DECLINING' ? 'down' as const : 'stable' as const,
    qualityScore: Math.round(sessionAnalytics?.averageQuality || 0),
  };

  const streakData = {
    currentStreak: userProfile?.streak || 0,
    longestStreak: userProfile?.streak || 0, // Could be added to user profile later
    streakHistory: Array.from({ length: 30 }, (_, i) => i < (userProfile?.streak || 0)), // Simplified - could be enhanced with actual history
    todayCompleted: sessionAnalytics?.totalSessions && sessionAnalytics.totalSessions > 0, // If any sessions completed today
  };

  // Ambient sound management
  const audioController = useAmbientSound({
    initialSound: ambientSoundType,
    initialVolume: volume / 100, // Convert percentage to 0-1 range
    autoLoad: soundEnabled,
    loop: true,
    fadeInDuration: 1000,
    fadeOutDuration: 1000,
  });

  // Sync audio state with local state
  useEffect(() => {
    if (audioController.volume !== volume / 100) {
      audioController.setVolume(volume / 100);
    }
  }, [volume, audioController]);

  useEffect(() => {
    if (audioController.currentSound !== ambientSoundType) {
      audioController.changeSound(ambientSoundType);
    }
  }, [ambientSoundType, audioController]);

  // Handle audio loading states and errors
  useEffect(() => {
    if (audioController.error) {
      console.error('Ambient sound error:', audioController.error);
    }
  }, [audioController.error]);

  // Timer-audio synchronization
  useEffect(() => {
    // Auto-start ambient sound when timer starts and sound is enabled
    if (timerLogic.isRunning && soundEnabled && ambientSoundType !== 'none' && !audioController.isPlaying) {
      audioController.playWithTimer(ambientSoundType, true);
    }

    // Auto-stop ambient sound when timer stops (if in timer mode)
    if (!timerLogic.isRunning && !timerLogic.isPaused && audioController.isPlaying) {
      audioController.stopWithTimer(true);
    }
  }, [timerLogic.isRunning, timerLogic.isPaused, soundEnabled, ambientSoundType, audioController]);

  // Handle session completion - stop ambient sound
  useEffect(() => {
    const originalOnSessionComplete = timerLogic.onSessionComplete;

    // Override session complete handler to stop ambient sound
    if (originalOnSessionComplete && typeof originalOnSessionComplete === 'function') {
      // Store the original handler and wrap it
      const wrappedHandler = (sessionId: string) => {
        // Stop ambient sound when session completes
        if (audioController.isPlaying) {
          audioController.stopWithTimer(true);
        }
        // Call original handler
        originalOnSessionComplete(sessionId);
      };

      // Replace the handler (this is a bit of a hack, but works for this integration)
      (timerLogic as any).onSessionComplete = wrappedHandler;
    }

    return () => {
      // Restore original handler on cleanup
      if (originalOnSessionComplete) {
        (timerLogic as any).onSessionComplete = originalOnSessionComplete;
      }
    };
  }, [timerLogic.onSessionComplete, audioController]);
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
          setShowTaskSelectionModal(true);
        }}
      />
      {taskIntegration.currentTask && (
        <TaskProgress
          task={taskIntegration.currentTask}
          sessionCount={timerLogic.sessionsCompleted}
          totalSessions={4}
          showTimeEstimate={!isMobile}
          isTimerRunning={timerLogic.isRunning}
          sessionType={timerLogic.sessionType}
          sessionProgress={timerLogic.progress}
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
          currentSound={ambientSoundType}
          focusMode={focusModeActive}
          notificationsEnabled={true}
          isPlaying={audioController.isPlaying}
          isLoading={audioController.isLoading}
          audioError={audioController.error}
          onSoundToggle={(enabled) => {
            setSoundEnabled(enabled);
            if (enabled && ambientSoundType !== 'none') {
              audioController.play(ambientSoundType);
            } else {
              audioController.stop(true);
            }
          }}
          onVolumeChange={(newVolume) => {
            setVolume(newVolume);
            audioController.setVolume(newVolume / 100);
          }}
          onSoundChange={(newSound) => {
            setAmbientSoundType(newSound as any);
            if (soundEnabled && newSound !== 'none') {
              audioController.changeSound(newSound as any);
            } else if (newSound === 'none') {
              audioController.stop(true);
            }
          }}
          onFocusModeToggle={setFocusModeActive}
          onNotificationsToggle={(enabled) => console.log('Notifications:', enabled)}
        />
      )}
    </div>
  );

  // Session statistics component
  const SessionStatsComponent = sessionAnalytics ? (
    <div style={{
      background: 'rgba(255, 255, 255, 0.8)',
      borderRadius: '12px',
      padding: '16px',
      border: '1px solid rgba(0, 0, 0, 0.1)'
    }}>
      <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: '600', color: '#2C3E50' }}>
        📊 Session Stats
      </h4>
      <div style={{ display: 'grid', gap: '8px', fontSize: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#7F8C8D' }}>Today's Sessions:</span>
          <span style={{ fontWeight: '600', color: '#2C3E50' }}>
            {sessionAnalytics.completedSessions}/{sessionAnalytics.totalSessions}
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#7F8C8D' }}>Focus Time:</span>
          <span style={{ fontWeight: '600', color: '#2C3E50' }}>
            {Math.round(sessionAnalytics.totalMinutes / 60)}h {sessionAnalytics.totalMinutes % 60}m
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#7F8C8D' }}>Average Quality:</span>
          <span style={{ fontWeight: '600', color: '#2C3E50' }}>
            {Math.round(sessionAnalytics.averageQuality * 100)}%
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#7F8C8D' }}>Completion Rate:</span>
          <span style={{ fontWeight: '600', color: '#27AE60' }}>
            {Math.round(sessionAnalytics.completionRate)}%
          </span>
        </div>
      </div>
    </div>
  ) : null;

  // Task analytics component
  const TaskAnalyticsComponent = taskAnalytics ? (
    <div style={{
      background: 'rgba(255, 255, 255, 0.8)',
      borderRadius: '12px',
      padding: '16px',
      border: '1px solid rgba(0, 0, 0, 0.1)'
    }}>
      <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: '600', color: '#2C3E50' }}>
        🎯 Task Progress
      </h4>
      <div style={{ display: 'grid', gap: '8px', fontSize: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#7F8C8D' }}>Completed Today:</span>
          <span style={{ fontWeight: '600', color: '#27AE60' }}>
            {taskAnalytics.completedTasks}/{taskAnalytics.totalTasks}
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#7F8C8D' }}>Overdue Tasks:</span>
          <span style={{ fontWeight: '600', color: taskAnalytics.overdueTasks > 0 ? '#E74C3C' : '#27AE60' }}>
            {taskAnalytics.overdueTasks}
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#7F8C8D' }}>Completion Rate:</span>
          <span style={{ fontWeight: '600', color: '#2C3E50' }}>
            {Math.round(taskAnalytics.completionRate)}%
          </span>
        </div>
      </div>
    </div>
  ) : null;

  // Analytics components
  const AnalyticsComponent = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <TodayProgress {...analyticsData} />
      {(isTablet || isDesktop) && (
        <div style={{ display: 'flex', gap: '16px', flexDirection: isDesktop ? 'row' : 'column' }}>
          {SessionStatsComponent}
          {TaskAnalyticsComponent}
        </div>
      )}
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
                setAmbientSound={setAmbientSoundType}
                currentSound={ambientSoundType}
                soundEnabled={soundEnabled}
              />
            </>
          )}
        </>
      )}
    </div>
  );

  const isLoading = wellnessData.isLoading ||
                   taskIntegration.isLoading ||
                   focusAnalyticsLoading ||
                   taskAnalyticsLoading ||
                   sessionAnalyticsLoading ||
                   wellnessAnalyticsLoading ||
                   audioController.isLoading;

  const hasError = focusAnalyticsError || taskAnalyticsError || sessionAnalyticsError || wellnessAnalyticsError;

  if (isLoading) {
    return (
      <TimerScreenContainer className={className}>
        <GlobalStyles />
        <LoadingOverlay>Loading Timer...</LoadingOverlay>
      </TimerScreenContainer>
    );
  }

  if (hasError) {
    console.error('Analytics loading error:', { focusAnalyticsError, taskAnalyticsError, sessionAnalyticsError, wellnessAnalyticsError });
    // Continue with default values if analytics fail to load
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

      {/* Task Selection Modal */}
      <TaskSelectionModal
        isOpen={showTaskSelectionModal}
        onClose={() => setShowTaskSelectionModal(false)}
        onTaskSelect={(taskId) => {
          taskIntegration.selectTask(taskId);
        }}
        tasks={taskIntegration.tasks}
        isLoading={taskIntegration.isLoading}
        error={taskIntegration.error}
      />
    </TimerScreenContainer>
  );
};

export type { NewTimerScreenProps };