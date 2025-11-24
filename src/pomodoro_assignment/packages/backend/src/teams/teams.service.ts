import {
  Injectable,
  NotFoundException,
  ForbiddenException,
  ConflictException,
  BadRequestException,
} from '@nestjs/common';
import { DatabaseService } from '../config/database.config';
import {
  Team,
  TeamMember,
  User,
  Task,
  Session,
  TeamChallenge,
} from '@prisma/client';
import {
  TeamDto,
  TeamMemberDto,
  CreateTeamDto,
  UpdateTeamDto,
  JoinTeamDto,
  UpdateMemberRoleDto,
  InviteMembersDto,
  TeamStatsDto,
} from './dto';

type TeamWithMembers = Team & {
  members: (TeamMember & { user: User })[];
  _count: {
    challenges: number;
  };
};

@Injectable()
export class TeamsService {
  constructor(private readonly prisma: DatabaseService) {}

  /**
   * Create a new team
   */
  async createTeam(createTeamDto: CreateTeamDto, userId: string): Promise<TeamDto> {
    const team = await this.prisma.team.create({
      data: {
        name: createTeamDto.name,
        description: createTeamDto.description,
        avatar: createTeamDto.avatar,
        ownerId: userId,
      },
      include: {
        members: {
          include: {
            user: {
              select: {
                id: true,
                firstName: true,
                lastName: true,
                email: true,
                avatar: true,
              },
            },
          },
        },
        _count: {
          select: {
            challenges: true,
          },
        },
      },
    });

    // Add creator as team owner member
    await this.prisma.teamMember.create({
      data: {
        userId,
        teamId: team.id,
        role: 'OWNER',
      },
    });

    // Fetch the team with members
    const teamWithMembers = await this.findTeamById(team.id, userId);
    return this.formatTeamDto(teamWithMembers as TeamWithMembers);
  }

  /**
   * Get all teams for the current user
   */
  async getUserTeams(userId: string): Promise<TeamDto[]> {
    const teams = await this.prisma.team.findMany({
      where: {
        members: {
          some: {
            userId,
          },
        },
      },
      include: {
        members: {
          include: {
            user: {
              select: {
                id: true,
                firstName: true,
                lastName: true,
                email: true,
                avatar: true,
              },
            },
          },
        },
        _count: {
          select: {
            challenges: true,
          },
        },
      },
      orderBy: {
        updatedAt: 'desc',
      },
    });

    return teams.map((team) => this.formatTeamDto(team as TeamWithMembers));
  }

  /**
   * Get team by ID (if user is member)
   */
  async findTeamById(teamId: string, userId: string): Promise<Team | null> {
    return this.prisma.team.findFirst({
      where: {
        id: teamId,
        members: {
          some: {
            userId,
          },
        },
      },
      include: {
        members: {
          include: {
            user: {
              select: {
                id: true,
                firstName: true,
                lastName: true,
                email: true,
                avatar: true,
              },
            },
          },
        },
        _count: {
          select: {
            challenges: true,
          },
        },
      },
    });
  }

  /**
   * Get team by ID for user
   */
  async getTeamById(teamId: string, userId: string): Promise<TeamDto> {
    const team = await this.findTeamById(teamId, userId);
    if (!team) {
      throw new NotFoundException('Team not found or access denied');
    }

    return this.formatTeamDto(team as TeamWithMembers);
  }

  /**
   * Update team (only owner/admin)
   */
  async updateTeam(
    teamId: string,
    updateTeamDto: UpdateTeamDto,
    userId: string,
  ): Promise<TeamDto> {
    const membership = await this.getMembership(teamId, userId);
    if (!membership || !['OWNER', 'ADMIN'].includes(membership.role)) {
      throw new ForbiddenException('Only team owners or admins can update the team');
    }

    const updatedTeam = await this.prisma.team.update({
      where: { id: teamId },
      data: updateTeamDto,
      include: {
        members: {
          include: {
            user: {
              select: {
                id: true,
                firstName: true,
                lastName: true,
                email: true,
                avatar: true,
              },
            },
          },
        },
        _count: {
          select: {
            challenges: true,
          },
        },
      },
    });

    return this.formatTeamDto(updatedTeam as TeamWithMembers);
  }

  /**
   * Join a team
   */
  async joinTeam(teamId: string, joinTeamDto: JoinTeamDto, userId: string): Promise<TeamDto> {
    // Check if team exists
    const team = await this.prisma.team.findUnique({
      where: { id: teamId },
    });

    if (!team) {
      throw new NotFoundException('Team not found');
    }

    // Check if user is already a member
    const existingMembership = await this.prisma.teamMember.findUnique({
      where: {
        userId_teamId: {
          userId,
          teamId,
        },
      },
    });

    if (existingMembership) {
      throw new ConflictException('User is already a member of this team');
    }

    // Add user as member
    await this.prisma.teamMember.create({
      data: {
        userId,
        teamId,
        role: 'MEMBER',
      },
    });

    const updatedTeam = await this.findTeamById(teamId, userId);
    return this.formatTeamDto(updatedTeam as TeamWithMembers);
  }

  /**
   * Leave a team
   */
  async leaveTeam(teamId: string, userId: string): Promise<void> {
    const membership = await this.getMembership(teamId, userId);
    if (!membership) {
      throw new NotFoundException('User is not a member of this team');
    }

    // Check if user is the owner
    const team = await this.prisma.team.findUnique({
      where: { id: teamId },
      select: { ownerId: true },
    });

    if (team?.ownerId === userId) {
      throw new ForbiddenException('Team owners cannot leave the team. Transfer ownership first.');
    }

    await this.prisma.teamMember.delete({
      where: {
        id: membership.id,
      },
    });
  }

  /**
   * Remove member from team (only owner/admin)
   */
  async removeMember(teamId: string, memberUserId: string, userId: string): Promise<void> {
    const membership = await this.getMembership(teamId, userId);
    if (!membership || !['OWNER', 'ADMIN'].includes(membership.role)) {
      throw new ForbiddenException('Only team owners or admins can remove members');
    }

    const targetMembership = await this.getMembership(teamId, memberUserId);
    if (!targetMembership) {
      throw new NotFoundException('User is not a member of this team');
    }

    // Cannot remove the owner
    if (targetMembership.role === 'OWNER') {
      throw new ForbiddenException('Cannot remove the team owner');
    }

    await this.prisma.teamMember.delete({
      where: {
        id: targetMembership.id,
      },
    });
  }

  /**
   * Update member role (only owner)
   */
  async updateMemberRole(
    teamId: string,
    memberUserId: string,
    updateRoleDto: UpdateMemberRoleDto,
    userId: string,
  ): Promise<TeamMemberDto> {
    const membership = await this.getMembership(teamId, userId);
    if (!membership || membership.role !== 'OWNER') {
      throw new ForbiddenException('Only team owners can update member roles');
    }

    const targetMembership = await this.getMembership(teamId, memberUserId);
    if (!targetMembership) {
      throw new NotFoundException('User is not a member of this team');
    }

    // Cannot change owner role of self
    if (targetMembership.userId === userId && updateRoleDto.role !== 'OWNER') {
      throw new BadRequestException('Cannot change your own owner role. Transfer ownership first.');
    }

    const updatedMembership = await this.prisma.teamMember.update({
      where: { id: targetMembership.id },
      data: { role: updateRoleDto.role },
      include: {
        user: {
          select: {
            id: true,
            firstName: true,
            lastName: true,
            email: true,
            avatar: true,
          },
        },
      },
    });

    return this.formatTeamMemberDto(updatedMembership);
  }

  /**
   * Get team statistics
   */
  async getTeamStats(teamId: string, userId: string): Promise<TeamStatsDto> {
    const membership = await this.getMembership(teamId, userId);
    if (!membership) {
      throw new NotFoundException('Team not found or access denied');
    }

    const oneWeekAgo = new Date();
    oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);

    const [
      totalMembers,
      activeMembersCount,
      totalFocusTime,
      tasksCompleted,
      activeChallenges,
      memberWellnessScores,
    ] = await Promise.all([
      // Total members
      this.prisma.teamMember.count({
        where: { teamId },
      }),

      // Active members this week (members with sessions)
      this.prisma.teamMember.count({
        where: {
          teamId,
          user: {
            sessions: {
              some: {
                startTime: {
                  gte: oneWeekAgo,
                },
              },
            },
          },
        },
      }),

      // Total focus time this week
      this.prisma.session.aggregate({
        where: {
          user: {
            teamMembers: {
              some: { teamId },
            },
          },
          startTime: {
            gte: oneWeekAgo,
          },
          completed: true,
        },
        _sum: {
          duration: true,
        },
      }),

      // Tasks completed this week
      this.prisma.task.count({
        where: {
          teamId,
          status: 'COMPLETED',
          updatedAt: {
            gte: oneWeekAgo,
          },
        },
      }),

      // Active challenges
      this.prisma.teamChallenge.count({
        where: {
          teamId,
          isActive: true,
          endDate: {
            gte: new Date(),
          },
        },
      }),

      // Member wellness scores (from user profiles)
      this.prisma.user.findMany({
        where: {
          teamMembers: {
            some: { teamId },
          },
        },
        select: {
          id: true,
          totalFocusTime: true,
          tasksCompleted: true,
          streak: true,
        },
      }),
    ]);

    // Calculate average wellness score (simplified calculation)
    const averageWellnessScore = memberWellnessScores.length > 0
      ? memberWellnessScores.reduce((sum, user) => {
          const wellnessScore = Math.min(100,
            (user.totalFocusTime / 60) * 0.5 + // Focus time contribution
            user.tasksCompleted * 2 + // Task completion contribution
            user.streak * 5 // Streak contribution
          );
          return sum + wellnessScore;
        }, 0) / memberWellnessScores.length
      : 0;

    return {
      totalMembers,
      activeMembersThisWeek: activeMembersCount,
      totalFocusTimeThisWeek: totalFocusTime._sum.duration || 0,
      tasksCompletedThisWeek: tasksCompleted,
      averageWellnessScore: Math.round(averageWellnessScore * 10) / 10,
      activeChallenges,
    };
  }

  /**
   * Delete team (only owner)
   */
  async deleteTeam(teamId: string, userId: string): Promise<void> {
    const team = await this.prisma.team.findUnique({
      where: { id: teamId },
      select: { ownerId: true },
    });

    if (!team) {
      throw new NotFoundException('Team not found');
    }

    if (team.ownerId !== userId) {
      throw new ForbiddenException('Only team owners can delete the team');
    }

    await this.prisma.team.delete({
      where: { id: teamId },
    });
  }

  /**
   * Helper: Get user membership in team
   */
  private async getMembership(teamId: string, userId: string): Promise<TeamMember | null> {
    return this.prisma.teamMember.findUnique({
      where: {
        userId_teamId: {
          userId,
          teamId,
        },
      },
    });
  }

  /**
   * Helper: Format team DTO
   */
  private formatTeamDto(team: TeamWithMembers): TeamDto {
    return {
      id: team.id,
      name: team.name,
      description: team.description || undefined,
      avatar: team.avatar || undefined,
      ownerId: team.ownerId,
      members: team.members.map(this.formatTeamMemberDto),
      challengesCount: team._count.challenges,
      createdAt: team.createdAt.toISOString(),
      updatedAt: team.updatedAt.toISOString(),
    };
  }

  /**
   * Helper: Format team member DTO
   */
  private formatTeamMemberDto(member: TeamMember & { user: any }): TeamMemberDto {
    return {
      id: member.id,
      userId: member.userId,
      user: member.user,
      role: member.role as 'OWNER' | 'ADMIN' | 'MEMBER',
      joinedAt: member.joinedAt.toISOString(),
    };
  }
}