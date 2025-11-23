import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsEmail, IsString, IsOptional, IsInt, Min, Max, IsObject } from 'class-validator';
import { Type } from 'class-transformer';

export class CreateUserDto {
  @ApiProperty({
    description: 'User email address',
    example: 'user@example.com',
  })
  @IsEmail({}, { message: 'Please provide a valid email address' })
  email: string;

  @ApiProperty({
    description: 'User first name',
    example: 'John',
  })
  @IsString({ message: 'First name must be a string' })
  firstName: string;

  @ApiProperty({
    description: 'User last name',
    example: 'Doe',
  })
  @IsString({ message: 'Last name must be a string' })
  lastName: string;

  @ApiPropertyOptional({
    description: 'User avatar URL',
    example: 'https://example.com/avatar.jpg',
  })
  @IsOptional()
  @IsString({ message: 'Avatar must be a string' })
  avatar?: string;

  @ApiPropertyOptional({
    description: 'User level',
    example: 1,
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
    example: 0,
    minimum: 0,
  })
  @IsOptional()
  @Type(() => Number)
  @IsInt({ message: 'XP must be an integer' })
  @Min(0, { message: 'XP cannot be negative' })
  xp?: number;

  @ApiPropertyOptional({
    description: 'User preferences',
    example: {
      theme: 'dark',
      notifications: true,
      pomodoroLength: 25,
    },
  })
  @IsOptional()
  @IsObject({ message: 'Preferences must be an object' })
  preferences?: Record<string, any>;

  
  @ApiPropertyOptional({
    description: 'Team ID to assign user to',
    example: 'team-123',
  })
  @IsOptional()
  @IsString({ message: 'Team ID must be a string' })
  teamId?: string;
}