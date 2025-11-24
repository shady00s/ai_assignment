import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsOptional, IsArray, IsEnum } from 'class-validator';

export class TeamRole {
  @ApiProperty({ enum: ['OWNER', 'ADMIN', 'MEMBER'], description: 'Team member role' })
  role: 'OWNER' | 'ADMIN' | 'MEMBER';
}

export class TeamMemberDto {
  @ApiProperty({ description: 'Team member ID' })
  id: string;

  @ApiProperty({ description: 'User ID' })
  userId: string;

  @ApiProperty({ description: 'User information' })
  user: {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
    avatar?: string;
  };

  @ApiProperty({
    description: 'Member role in team',
    enum: ['OWNER', 'ADMIN', 'MEMBER'],
    example: 'MEMBER'
  })
  role: 'OWNER' | 'ADMIN' | 'MEMBER';

  @ApiProperty({
    description: 'When member joined the team',
    example: '2024-01-15T10:00:00.000Z'
  })
  joinedAt: string;
}

export class TeamDto {
  @ApiProperty({ description: 'Team ID' })
  id: string;

  @ApiProperty({ description: 'Team name', example: 'Development Team' })
  name: string;

  @ApiProperty({
    description: 'Team description',
    required: false,
    example: 'Main development team for OptoPomodoro'
  })
  description?: string;

  @ApiProperty({
    description: 'Team avatar URL',
    required: false,
    example: 'https://example.com/team-avatar.png'
  })
  avatar?: string;

  @ApiProperty({ description: 'Team owner ID' })
  ownerId: string;

  @ApiProperty({ description: 'Team members', type: [TeamMemberDto] })
  members: TeamMemberDto[];

  @ApiProperty({ description: 'Team challenges count', example: 5 })
  challengesCount?: number;

  @ApiProperty({
    description: 'Team creation date',
    example: '2024-01-15T10:00:00.000Z'
  })
  createdAt: string;

  @ApiProperty({
    description: 'Team last update date',
    example: '2024-01-20T15:30:00.000Z'
  })
  updatedAt: string;
}

export class CreateTeamDto {
  @ApiProperty({ description: 'Team name', example: 'Development Team' })
  @IsString()
  name: string;

  @ApiProperty({
    description: 'Team description',
    required: false,
    example: 'Main development team for OptoPomodoro'
  })
  @IsOptional()
  @IsString()
  description?: string;

  @ApiProperty({
    description: 'Team avatar URL',
    required: false,
    example: 'https://example.com/team-avatar.png'
  })
  @IsOptional()
  @IsString()
  avatar?: string;
}

export class UpdateTeamDto {
  @ApiProperty({
    description: 'Team name',
    required: false,
    example: 'Updated Development Team'
  })
  @IsOptional()
  @IsString()
  name?: string;

  @ApiProperty({
    description: 'Team description',
    required: false,
    example: 'Updated description for the team'
  })
  @IsOptional()
  @IsString()
  description?: string;

  @ApiProperty({
    description: 'Team avatar URL',
    required: false,
    example: 'https://example.com/new-avatar.png'
  })
  @IsOptional()
  @IsString()
  avatar?: string;
}

export class JoinTeamDto {
  @ApiProperty({
    description: 'Team invite code (optional)',
    required: false,
    example: 'INVITE123'
  })
  @IsOptional()
  @IsString()
  inviteCode?: string;
}

export class UpdateMemberRoleDto {
  @ApiProperty({
    description: 'New role for team member',
    enum: ['OWNER', 'ADMIN', 'MEMBER'],
    example: 'ADMIN'
  })
  @IsEnum(['OWNER', 'ADMIN', 'MEMBER'])
  role: 'OWNER' | 'ADMIN' | 'MEMBER';
}

export class InviteMembersDto {
  @ApiProperty({
    description: 'Email addresses to invite',
    example: ['user1@example.com', 'user2@example.com']
  })
  @IsArray()
  @IsString({ each: true })
  emails: string[];
}

export class TeamStatsDto {
  @ApiProperty({ description: 'Total members in team', example: 8 })
  totalMembers: number;

  @ApiProperty({ description: 'Active members this week', example: 6 })
  activeMembersThisWeek: number;

  @ApiProperty({ description: 'Total focus time this week (minutes)', example: 2400 })
  totalFocusTimeThisWeek: number;

  @ApiProperty({ description: 'Tasks completed this week', example: 45 })
  tasksCompletedThisWeek: number;

  @ApiProperty({ description: 'Average wellness score', example: 85.5 })
  averageWellnessScore: number;

  @ApiProperty({ description: 'Active challenges', example: 3 })
  activeChallenges: number;
}