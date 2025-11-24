import { ApiProperty } from '@nestjs/swagger';

export class TeamMemberUserDto {
  @ApiProperty({ description: 'User ID' })
  id: string;

  @ApiProperty({ description: 'First name' })
  firstName: string;

  @ApiProperty({ description: 'Last name' })
  lastName: string;

  @ApiProperty({ description: 'Email address' })
  email: string;

  @ApiProperty({ description: 'Avatar URL', required: false })
  avatar?: string;

  @ApiProperty({ description: 'Current wellness score', example: 85 })
  wellnessScore: number;

  @ApiProperty({ description: 'Current level', example: 5 })
  level: number;

  @ApiProperty({ description: 'Total XP points', example: 2500 })
  xp: number;

  @ApiProperty({ description: 'Current streak days', example: 7 })
  streak: number;
}

export class TeamMemberStatsDto {
  @ApiProperty({ description: 'User ID' })
  userId: string;

  @ApiProperty({ description: 'User information', type: TeamMemberUserDto })
  user: TeamMemberUserDto;

  @ApiProperty({
    description: 'Focus time in minutes for the period',
    example: 450,
    minimum: 0
  })
  focusTime: number;

  @ApiProperty({
    description: 'Number of tasks completed',
    example: 12,
    minimum: 0
  })
  tasksCompleted: number;

  @ApiProperty({
    description: 'Task completion rate percentage',
    example: 75,
    minimum: 0,
    maximum: 100
  })
  completionRate: number;

  @ApiProperty({
    description: 'Wellness score',
    example: 85,
    minimum: 0,
    maximum: 100
  })
  wellnessScore: number;

  @ApiProperty({
    description: 'Current streak days',
    example: 7,
    minimum: 0
  })
  streakDays: number;
}

export class TeamAnalyticsDto {
  @ApiProperty({ description: 'Team ID' })
  teamId: string;

  @ApiProperty({ description: 'Team name' })
  teamName: string;

  @ApiProperty({
    description: 'Number of team members',
    example: 8,
    minimum: 1
  })
  memberCount: number;

  @ApiProperty({
    description: 'Total focus time for team (minutes)',
    example: 3600,
    minimum: 0
  })
  totalFocusTime: number;

  @ApiProperty({
    description: 'Average focus time per member (minutes)',
    example: 450,
    minimum: 0
  })
  averageFocusTime: number;

  @ApiProperty({
    description: 'Total tasks completed by team',
    example: 96,
    minimum: 0
  })
  tasksCompleted: number;

  @ApiProperty({
    description: 'Average completion rate percentage',
    example: 80,
    minimum: 0,
    maximum: 100
  })
  averageCompletionRate: number;

  @ApiProperty({
    description: 'Top performing members',
    type: [TeamMemberStatsDto]
  })
  topPerformers: TeamMemberStatsDto[];

  @ApiProperty({
    description: 'Team focus trend',
    enum: ['IMPROVING', 'DECLINING', 'STABLE']
  })
  focusTrend: 'IMPROVING' | 'DECLINING' | 'STABLE';

  @ApiProperty({
    description: 'Team average wellness score',
    example: 82,
    minimum: 0,
    maximum: 100
  })
  wellnessScore: number;

  @ApiProperty({
    description: 'Team collaboration score',
    example: 75,
    minimum: 0,
    maximum: 100
  })
  collaborationScore: number;

  @ApiProperty({ description: 'Analytics period' })
  period: {
    startDate: string;
    endDate: string;
  };
}