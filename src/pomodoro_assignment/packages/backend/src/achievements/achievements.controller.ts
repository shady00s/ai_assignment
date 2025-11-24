import {
  Controller,
  Get,
  Post,
  Param,
  Body,
  UseGuards,
  Request,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiParam, ApiBearerAuth } from '@nestjs/swagger';
import { AchievementsService } from './achievements.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import {
  AchievementDto,
  UserAchievementDto,
  UnlockAchievementDto,
} from './dto';

@ApiTags('achievements')
@Controller('achievements')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class AchievementsController {
  constructor(private readonly achievementsService: AchievementsService) {}

  @Get()
  @ApiOperation({ summary: 'Get all available achievements' })
  @ApiResponse({
    status: 200,
    description: 'Achievements retrieved successfully',
    type: [AchievementDto],
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getAllAchievements() {
    return this.achievementsService.getAllAchievements();
  }

  @Get('user/stats')
  @ApiOperation({ summary: 'Get current user achievement statistics' })
  @ApiResponse({
    status: 200,
    description: 'User achievement statistics retrieved successfully',
    schema: {
      type: 'object',
      properties: {
        totalAchievements: { type: 'number', example: 50 },
        unlockedAchievements: { type: 'number', example: 12 },
        completionRate: { type: 'number', example: 24 },
        totalXpFromAchievements: { type: 'number', example: 600 },
        recentUnlocks: { type: 'array', items: { $ref: '#/components/schemas/UserAchievementDto' } },
      },
    },
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getUserAchievementStats(@Request() req) {
    return this.achievementsService.getUserAchievementStats(req.user.id);
  }

  @Get('user/progress')
  @ApiOperation({ summary: 'Update and get user achievement progress' })
  @ApiResponse({
    status: 200,
    description: 'Achievement progress updated and retrieved successfully',
    type: [UserAchievementDto],
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async updateUserAchievementProgress(@Request() req) {
    return this.achievementsService.updateAchievementProgress(req.user.id);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get user achievement by ID' })
  @ApiParam({
    name: 'id',
    description: 'User achievement ID',
    example: 'user-achievement-123',
  })
  @ApiResponse({
    status: 200,
    description: 'User achievement retrieved successfully',
    type: UserAchievementDto,
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 404,
    description: 'User achievement not found',
  })
  async getUserAchievement(@Param('id') id: string, @Request() req) {
    return this.achievementsService.getUserAchievementById(id, req.user.id);
  }

  @Post(':achievementId/unlock')
  @ApiOperation({ summary: 'Unlock an achievement for the current user' })
  @ApiParam({
    name: 'achievementId',
    description: 'Achievement ID to unlock',
    example: 'achievement-123',
  })
  @ApiResponse({
    status: 200,
    description: 'Achievement unlocked successfully',
    type: UserAchievementDto,
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 404,
    description: 'Achievement not found or inactive',
  })
  @ApiResponse({
    status: 409,
    description: 'Achievement already unlocked',
  })
  async unlockAchievement(
    @Param('achievementId') achievementId: string,
    @Body() unlockDto: UnlockAchievementDto,
    @Request() req,
  ) {
    const unlockedAchievement = await this.achievementsService.unlockAchievement(
      req.user.id,
      achievementId,
    );

    if (!unlockedAchievement) {
      return { message: 'Achievement already unlocked' };
    }

    return unlockedAchievement;
  }
}