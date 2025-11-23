import { ApiPropertyOptional } from '@nestjs/swagger';
import { IsEmail, IsString, IsOptional, IsInt, Min, Max, IsObject } from 'class-validator';
import { Type } from 'class-transformer';
import { CreateUserDto } from './create-user.dto';

export class UpdateUserDto {
  @ApiPropertyOptional({
    description: 'User email address',
    example: 'newemail@example.com',
  })
  @IsOptional()
  @IsEmail({}, { message: 'Please provide a valid email address' })
  email?: string;

  @ApiPropertyOptional({
    description: 'User first name',
    example: 'John',
  })
  @IsOptional()
  @IsString({ message: 'First name must be a string' })
  firstName?: string;

  @ApiPropertyOptional({
    description: 'User last name',
    example: 'Doe',
  })
  @IsOptional()
  @IsString({ message: 'Last name must be a string' })
  lastName?: string;

  @ApiPropertyOptional({
    description: 'User avatar URL',
    example: 'https://example.com/new-avatar.jpg',
  })
  @IsOptional()
  @IsString({ message: 'Avatar must be a string' })
  avatar?: string;

  @ApiPropertyOptional({
    description: 'User level',
    example: 5,
    minimum: 1,
    maximum: 100,
  })
  @IsOptional()
  @Type(() => Number)
  @IsInt({ message: 'Level must be an integer' })
  @Min(1, { message: 'Level must be at least 1' })
  @Max(100, { message: 'Level cannot exceed 100' })
  level?: number;

  @ApiPropertyOptional({
    description: 'User experience points',
    example: 500,
    minimum: 0,
  })
  @IsOptional()
  @Type(() => Number)
  @IsInt({ message: 'XP must be an integer' })
  @Min(0, { message: 'XP cannot be negative' })
  xp?: number;

  @ApiPropertyOptional({
    description: 'User current streak',
    example: 7,
    minimum: 0,
  })
  @IsOptional()
  @Type(() => Number)
  @IsInt({ message: 'Streak must be an integer' })
  @Min(0, { message: 'Streak cannot be negative' })
  streak?: number;

  @ApiPropertyOptional({
    description: 'Total focus time in minutes',
    example: 1200,
    minimum: 0,
  })
  @IsOptional()
  @Type(() => Number)
  @IsInt({ message: 'Total focus time must be an integer' })
  @Min(0, { message: 'Total focus time cannot be negative' })
  totalFocusTime?: number;

  @ApiPropertyOptional({
    description: 'Number of completed tasks',
    example: 45,
    minimum: 0,
  })
  @IsOptional()
  @Type(() => Number)
  @IsInt({ message: 'Tasks completed must be an integer' })
  @Min(0, { message: 'Tasks completed cannot be negative' })
  tasksCompleted?: number;

  @ApiPropertyOptional({
    description: 'User quality score',
    example: 4.5,
    minimum: 0,
    maximum: 5,
  })
  @IsOptional()
  @Type(() => Number)
  @Min(0, { message: 'Quality score cannot be negative' })
  @Max(5, { message: 'Quality score cannot exceed 5' })
  qualityScore?: number;

  @ApiPropertyOptional({
    description: 'User wellness score',
    example: 8.2,
    minimum: 0,
    maximum: 10,
  })
  @IsOptional()
  @Type(() => Number)
  @Min(0, { message: 'Wellness score cannot be negative' })
  @Max(10, { message: 'Wellness score cannot exceed 10' })
  wellnessScore?: number;

  @ApiPropertyOptional({
    description: 'User preferences',
    example: {
      theme: 'light',
      notifications: false,
      pomodoroLength: 30,
    },
  })
  @IsOptional()
  @IsObject({ message: 'Preferences must be an object' })
  preferences?: Record<string, any>;

  
  @ApiPropertyOptional({
    description: 'Team ID to assign user to',
    example: 'team-456',
  })
  @IsOptional()
  @IsString({ message: 'Team ID must be a string' })
  teamId?: string;
}