/**
 * Utility functions for formatting dashboard data
 */

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

export type TrendType = 'IMPROVING' | 'DECLINING' | 'STABLE';

export const getTrendColor = (trend: TrendType): string => {
  switch (trend) {
    case 'IMPROVING': return '#7FA870'; // Success green
    case 'DECLINING': return '#C85A5A'; // Error red
    case 'STABLE': return '#8B7D7B'; // Neutral gray
    default: return '#8B7D7B';
  }
};

export const getTrendIcon = (trend: TrendType): string => {
  switch (trend) {
    case 'IMPROVING': return '📈';
    case 'DECLINING': return '📉';
    case 'STABLE': return '➡️';
    default: return '➡️';
  }
};

export const getTrendLabel = (trend: TrendType): string => {
  switch (trend) {
    case 'IMPROVING': return 'Improving';
    case 'DECLINING': return 'Declining';
    case 'STABLE': return 'Stable';
    default: return 'Stable';
  }
};

export const formatMoodRating = (rating: number): string => {
  const emojis = ['😔', '😐', '🙂', '😊', '😄'];
  return emojis[Math.max(0, Math.min(4, rating - 1))] || '😐';
};

export const formatStressLevel = (level: number): string => {
  const emojis = ['😌', '😊', '😐', '😰', '😓'];
  // Invert for better UX (lower stress is better)
  const invertedLevel = 6 - level;
  return emojis[Math.max(0, Math.min(4, invertedLevel - 1))] || '😐';
};

export const formatEnergyLevel = (level: number): string => {
  const emojis = ['😴', '🥱', '😐', '⚡', '🔥'];
  return emojis[Math.max(0, Math.min(4, level - 1))] || '😐';
};

export const getLevelColor = (level: number): string => {
  if (level >= 20) return '#FFD700'; // Gold
  if (level >= 15) return '#C0C0C0'; // Silver
  if (level >= 10) return '#CD7F32'; // Bronze
  if (level >= 5) return '#7A8B7F';   // Moss Green
  return '#8B7D7B';                  // Gray
};

export const getXpProgress = (currentXp: number, level: number): { current: number; total: number; percentage: number } => {
  const xpInCurrentLevel = currentXp - ((level - 1) * 100);
  const percentage = Math.min(100, Math.max(0, Math.round((xpInCurrentLevel / 100) * 100)));

  return {
    current: xpInCurrentLevel,
    total: 100,
    percentage
  };
};

export const getStreakMessage = (streak: number): string => {
  if (streak === 0) return 'Start your streak today!';
  if (streak === 1) return 'Great start! Keep it going!';
  if (streak <= 3) return `${streak} days! You're building momentum!`;
  if (streak <= 7) return `${streak} days! Amazing consistency!`;
  if (streak <= 14) return `${streak} days! You're on fire! 🔥`;
  if (streak <= 30) return `${streak} days! Legendary dedication!`;
  return `${streak} days! You're a focus master! 🏆`;
};