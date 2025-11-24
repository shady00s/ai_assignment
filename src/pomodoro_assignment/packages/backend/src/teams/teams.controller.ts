import {
  Controller,
  Get,
  Post,
  Patch,
  Delete,
  Body,
  Param,
  Query,
  UseGuards,
  Request,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiParam, ApiBearerAuth } from '@nestjs/swagger';
import { TeamsService } from './teams.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import {
  TeamDto,
  CreateTeamDto,
  UpdateTeamDto,
  JoinTeamDto,
  UpdateMemberRoleDto,
  TeamStatsDto,
} from './dto';

@ApiTags('teams')
@Controller('teams')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class TeamsController {
  constructor(private readonly teamsService: TeamsService) {}

  @Post()
  @ApiOperation({ summary: 'Create a new team' })
  @ApiResponse({
    status: 201,
    description: 'Team created successfully',
    type: TeamDto,
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async createTeam(
    @Body() createTeamDto: CreateTeamDto,
    @Request() req,
  ): Promise<TeamDto> {
    return this.teamsService.createTeam(createTeamDto, req.user.id);
  }

  @Get()
  @ApiOperation({ summary: 'Get all teams for the current user' })
  @ApiResponse({
    status: 200,
    description: 'Teams retrieved successfully',
    type: [TeamDto],
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getTeams(@Request() req): Promise<TeamDto[]> {
    return this.teamsService.getUserTeams(req.user.id);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get team by ID' })
  @ApiParam({
    name: 'id',
    description: 'Team ID',
    example: 'team-123',
  })
  @ApiResponse({
    status: 200,
    description: 'Team retrieved successfully',
    type: TeamDto,
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 404,
    description: 'Team not found or access denied',
  })
  async getTeam(@Param('id') id: string, @Request() req): Promise<TeamDto> {
    return this.teamsService.getTeamById(id, req.user.id);
  }

  @Patch(':id')
  @ApiOperation({ summary: 'Update team information' })
  @ApiParam({
    name: 'id',
    description: 'Team ID',
    example: 'team-123',
  })
  @ApiResponse({
    status: 200,
    description: 'Team updated successfully',
    type: TeamDto,
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 403,
    description: 'Forbidden - only owners or admins can update team',
  })
  @ApiResponse({
    status: 404,
    description: 'Team not found or access denied',
  })
  async updateTeam(
    @Param('id') id: string,
    @Body() updateTeamDto: UpdateTeamDto,
    @Request() req,
  ): Promise<TeamDto> {
    return this.teamsService.updateTeam(id, updateTeamDto, req.user.id);
  }

  @Post(':id/join')
  @ApiOperation({ summary: 'Join a team' })
  @ApiParam({
    name: 'id',
    description: 'Team ID to join',
    example: 'team-123',
  })
  @ApiResponse({
    status: 200,
    description: 'Successfully joined the team',
    type: TeamDto,
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 404,
    description: 'Team not found',
  })
  @ApiResponse({
    status: 409,
    description: 'User is already a member of this team',
  })
  async joinTeam(
    @Param('id') id: string,
    @Body() joinTeamDto: JoinTeamDto,
    @Request() req,
  ): Promise<TeamDto> {
    return this.teamsService.joinTeam(id, joinTeamDto, req.user.id);
  }

  @Post(':id/leave')
  @ApiOperation({ summary: 'Leave a team' })
  @ApiParam({
    name: 'id',
    description: 'Team ID to leave',
    example: 'team-123',
  })
  @ApiResponse({
    status: 200,
    description: 'Successfully left the team',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 403,
    description: 'Forbidden - team owners cannot leave the team',
  })
  @ApiResponse({
    status: 404,
    description: 'User is not a member of this team',
  })
  async leaveTeam(@Param('id') id: string, @Request() req): Promise<void> {
    return this.teamsService.leaveTeam(id, req.user.id);
  }

  @Delete(':id/members/:memberId')
  @ApiOperation({ summary: 'Remove a member from the team' })
  @ApiParam({
    name: 'id',
    description: 'Team ID',
    example: 'team-123',
  })
  @ApiParam({
    name: 'memberId',
    description: 'User ID of member to remove',
    example: 'user-456',
  })
  @ApiResponse({
    status: 200,
    description: 'Member removed successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 403,
    description: 'Forbidden - only owners or admins can remove members',
  })
  @ApiResponse({
    status: 404,
    description: 'Team or member not found',
  })
  async removeMember(
    @Param('id') id: string,
    @Param('memberId') memberId: string,
    @Request() req,
  ): Promise<void> {
    return this.teamsService.removeMember(id, memberId, req.user.id);
  }

  @Patch(':id/members/:memberId/role')
  @ApiOperation({ summary: 'Update member role' })
  @ApiParam({
    name: 'id',
    description: 'Team ID',
    example: 'team-123',
  })
  @ApiParam({
    name: 'memberId',
    description: 'User ID of member to update',
    example: 'user-456',
  })
  @ApiResponse({
    status: 200,
    description: 'Member role updated successfully',
    type: TeamDto,
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 403,
    description: 'Forbidden - only team owners can update member roles',
  })
  @ApiResponse({
    status: 404,
    description: 'Team or member not found',
  })
  async updateMemberRole(
    @Param('id') id: string,
    @Param('memberId') memberId: string,
    @Body() updateRoleDto: UpdateMemberRoleDto,
    @Request() req,
  ): Promise<any> {
    return this.teamsService.updateMemberRole(id, memberId, updateRoleDto, req.user.id);
  }

  @Get(':id/stats')
  @ApiOperation({ summary: 'Get team statistics' })
  @ApiParam({
    name: 'id',
    description: 'Team ID',
    example: 'team-123',
  })
  @ApiResponse({
    status: 200,
    description: 'Team statistics retrieved successfully',
    type: TeamStatsDto,
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 404,
    description: 'Team not found or access denied',
  })
  async getTeamStats(@Param('id') id: string, @Request() req): Promise<TeamStatsDto> {
    return this.teamsService.getTeamStats(id, req.user.id);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Delete a team' })
  @ApiParam({
    name: 'id',
    description: 'Team ID to delete',
    example: 'team-123',
  })
  @ApiResponse({
    status: 200,
    description: 'Team deleted successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 403,
    description: 'Forbidden - only team owners can delete teams',
  })
  @ApiResponse({
    status: 404,
    description: 'Team not found',
  })
  async deleteTeam(@Param('id') id: string, @Request() req): Promise<void> {
    return this.teamsService.deleteTeam(id, req.user.id);
  }
}