import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  Query,
  Put,
  UseGuards,
  HttpCode,
  HttpStatus,
  ParseIntPipe,
  DefaultValuePipe,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiParam, ApiQuery, ApiBearerAuth } from '@nestjs/swagger';
import { ThrottlerGuard } from '@nestjs/throttler';
import { UsersService } from './users.service';
import { CreateUserDto, UpdateUserDto, UpdatePreferencesDto, UpdateProfileDto } from './dto';
import { AuthGuard } from '../auth/guards/auth.guard';
import { CurrentUser } from '../auth/decorators/current-user.decorator';
import { LoggerService } from '../core/logger/logger.service';

@ApiTags('Users')
@Controller('users')
@UseGuards(ThrottlerGuard)
export class UsersController {
  constructor(
    private readonly usersService: UsersService,
    private readonly logger: LoggerService,
  ) {}

  @Post()
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @HttpCode(HttpStatus.CREATED)
  @ApiOperation({ summary: 'Create a new user' })
  @ApiResponse({
    status: 201,
    description: 'User successfully created',
    type: CreateUserDto,
  })
  @ApiResponse({
    status: 409,
    description: 'Email already exists',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async create(@Body() createUserDto: CreateUserDto, @CurrentUser() currentUser: any) {
    this.logger.logUserAction('USER_CREATION_ATTEMPT', currentUser.userId, {
      targetEmail: createUserDto.email,
    });

    return this.usersService.create(createUserDto);
  }

  @Get()
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get all users with pagination and filtering' })
  @ApiResponse({
    status: 200,
    description: 'Users retrieved successfully',
  })
  @ApiQuery({
    name: 'page',
    required: false,
    type: Number,
    description: 'Page number for pagination',
  })
  @ApiQuery({
    name: 'limit',
    required: false,
    type: Number,
    description: 'Number of users per page',
  })
  @ApiQuery({
    name: 'teamId',
    required: false,
    type: String,
    description: 'Filter users by team ID',
  })
  @ApiQuery({
    name: 'search',
    required: false,
    type: String,
    description: 'Search users by name or email',
  })
  async findAll(
    @Query('page', new DefaultValuePipe(1), ParseIntPipe) page: number,
    @Query('limit', new DefaultValuePipe(20), ParseIntPipe) limit: number,
    @Query('teamId') teamId?: string,
    @Query('search') search?: string,
    @CurrentUser() currentUser?: any,
  ) {
    this.logger.log('Fetching users list', 'UsersController', {
      page,
      limit,
      teamId,
      search,
      requestedBy: currentUser?.userId,
    });

    return this.usersService.findAll({ page, limit, teamId, search });
  }

  @Get('me')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get current user profile' })
  @ApiResponse({
    status: 200,
    description: 'Current user profile retrieved successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getCurrentUser(@CurrentUser() currentUser: any) {
    this.logger.logUserAction('PROFILE_ACCESSED', currentUser.userId);

    return this.usersService.findById(currentUser.userId);
  }

  @Get('profile')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get current user profile (frontend compatibility)' })
  @ApiResponse({
    status: 200,
    description: 'Current user profile retrieved successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getProfile(@CurrentUser() currentUser: any) {
    this.logger.logUserAction('PROFILE_ACCESSED', currentUser.userId);

    return this.usersService.findById(currentUser.userId);
  }

  @Get('me/statistics')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get current user statistics and analytics' })
  @ApiResponse({
    status: 200,
    description: 'User statistics retrieved successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getMyStatistics(@CurrentUser() currentUser: any) {
    this.logger.logUserAction('STATISTICS_ACCESSED', currentUser.userId);

    return this.usersService.getUserStatistics(currentUser.userId);
  }

  @Get(':id')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get user by ID' })
  @ApiParam({
    name: 'id',
    description: 'User ID',
  })
  @ApiResponse({
    status: 200,
    description: 'User retrieved successfully',
  })
  @ApiResponse({
    status: 404,
    description: 'User not found',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async findOne(@Param('id') id: string, @CurrentUser() currentUser: any) {
    this.logger.log('Fetching user details', 'UsersController', {
      targetUserId: id,
      requestedBy: currentUser.userId,
    });

    return this.usersService.findById(id);
  }

  @Get(':id/statistics')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get user statistics and analytics' })
  @ApiParam({
    name: 'id',
    description: 'User ID',
  })
  @ApiResponse({
    status: 200,
    description: 'User statistics retrieved successfully',
  })
  @ApiResponse({
    status: 404,
    description: 'User not found',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async getUserStatistics(@Param('id') id: string, @CurrentUser() currentUser: any) {
    this.logger.log('Fetching user statistics', 'UsersController', {
      targetUserId: id,
      requestedBy: currentUser.userId,
    });

    return this.usersService.getUserStatistics(id);
  }

  @Get('email/:email')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get user by email' })
  @ApiParam({
    name: 'email',
    description: 'User email',
  })
  @ApiResponse({
    status: 200,
    description: 'User retrieved successfully',
  })
  @ApiResponse({
    status: 404,
    description: 'User not found',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async findByEmail(@Param('email') email: string, @CurrentUser() currentUser: any) {
    this.logger.log('Fetching user by email', 'UsersController', {
      targetEmail: email,
      requestedBy: currentUser.userId,
    });

    return this.usersService.findByEmail(email);
  }

  @Patch('me')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update current user profile' })
  @ApiResponse({
    status: 200,
    description: 'Current user profile updated successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 409,
    description: 'Email already in use',
  })
  async updateMyProfile(
    @Body() updateUserDto: UpdateUserDto,
    @CurrentUser() currentUser: any,
  ) {
    this.logger.logUserAction('PROFILE_UPDATE_ATTEMPT', currentUser.userId, {
      updateFields: Object.keys(updateUserDto),
    });

    return this.usersService.update(currentUser.userId, updateUserDto);
  }

  @Put('profile')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update current user profile (frontend compatibility)' })
  @ApiResponse({
    status: 200,
    description: 'Current user profile updated successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 409,
    description: 'Email already in use',
  })
  async updateProfile(
    @Body() updateProfileDto: UpdateProfileDto,
    @CurrentUser() currentUser: any,
  ) {
    this.logger.logUserAction('PROFILE_UPDATE_ATTEMPT', currentUser.userId, {
      updateFields: Object.keys(updateProfileDto),
    });

    // Extract preferences from the nested structure
    const preferences = updateProfileDto.preferences;

    // Use updatePreferences method with PUT semantics (complete replacement)
    return this.usersService.updatePreferences(currentUser.userId, preferences, true);
  }

  @Patch('profile')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update current user profile info (PATCH for user data)' })
  @ApiResponse({
    status: 200,
    description: 'Current user profile info updated successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 409,
    description: 'Email already in use',
  })
  async patchProfile(
    @Body() updateUserDto: UpdateUserDto,
    @CurrentUser() currentUser: any,
  ) {
    this.logger.logUserAction('PROFILE_PATCH_ATTEMPT', currentUser.userId, {
      updateFields: Object.keys(updateUserDto),
    });

    return this.usersService.update(currentUser.userId, updateUserDto);
  }

  @Patch('preferences')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update user preferences' })
  @ApiResponse({
    status: 200,
    description: 'User preferences updated successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async updatePreferences(
    @Body() updatePreferencesDto: UpdatePreferencesDto,
    @CurrentUser() currentUser: any,
  ) {
    this.logger.logUserAction('PREFERENCES_UPDATE_ATTEMPT', currentUser.userId, {
      updateFields: Object.keys(updatePreferencesDto),
    });

    return this.usersService.updatePreferences(currentUser.userId, updatePreferencesDto);
  }

  @Patch(':id')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update user by ID' })
  @ApiParam({
    name: 'id',
    description: 'User ID',
  })
  @ApiResponse({
    status: 200,
    description: 'User updated successfully',
  })
  @ApiResponse({
    status: 404,
    description: 'User not found',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  @ApiResponse({
    status: 409,
    description: 'Email already in use',
  })
  async update(
    @Param('id') id: string,
    @Body() updateUserDto: UpdateUserDto,
    @CurrentUser() currentUser: any,
  ) {
    this.logger.logUserAction('USER_UPDATE_ATTEMPT', currentUser.userId, {
      targetUserId: id,
      updateFields: Object.keys(updateUserDto),
    });

    return this.usersService.update(id, updateUserDto);
  }

  @Patch(':id/level')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Update user level and XP' })
  @ApiParam({
    name: 'id',
    description: 'User ID',
  })
  @ApiResponse({
    status: 200,
    description: 'User level updated successfully',
  })
  @ApiResponse({
    status: 404,
    description: 'User not found',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async updateUserLevel(
    @Param('id') id: string,
    @Body('xpGained', ParseIntPipe) xpGained: number,
    @CurrentUser() currentUser: any,
  ) {
    this.logger.logUserAction('USER_LEVEL_UPDATE_ATTEMPT', currentUser.userId, {
      targetUserId: id,
      xpGained,
    });

    return this.usersService.updateUserLevel(id, xpGained);
  }

  @Delete('me')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiOperation({ summary: 'Delete current user account' })
  @ApiResponse({
    status: 204,
    description: 'Current user account deleted successfully',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async removeMyAccount(@CurrentUser() currentUser: any) {
    this.logger.logUserAction('ACCOUNT_DELETION_ATTEMPT', currentUser.userId);

    await this.usersService.remove(currentUser.userId);
  }

  @Delete(':id')
  @UseGuards(AuthGuard)
  @ApiBearerAuth()
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiOperation({ summary: 'Delete user by ID' })
  @ApiParam({
    name: 'id',
    description: 'User ID',
  })
  @ApiResponse({
    status: 204,
    description: 'User deleted successfully',
  })
  @ApiResponse({
    status: 404,
    description: 'User not found',
  })
  @ApiResponse({
    status: 401,
    description: 'Unauthorized',
  })
  async remove(@Param('id') id: string, @CurrentUser() currentUser: any) {
    this.logger.logUserAction('USER_DELETION_ATTEMPT', currentUser.userId, {
      targetUserId: id,
    });

    await this.usersService.remove(id);
  }
}