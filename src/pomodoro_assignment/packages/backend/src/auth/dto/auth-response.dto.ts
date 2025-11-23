import { ApiProperty } from '@nestjs/swagger';

export class AuthResponseDto {
  @ApiProperty({
    description: 'Authentication token',
    example: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
  })
  token: string;

  @ApiProperty({
    description: 'Refresh token for getting new access tokens',
    example: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
  })
  refreshToken: string;

  @ApiProperty({
    description: 'User information',
    type: 'object',
    properties: {
      id: { description: 'User ID', example: 'user-123', type: 'string' },
      email: { description: 'User email', example: 'user@example.com', type: 'string' },
      firstName: { description: 'User first name', example: 'John', type: 'string' },
      lastName: { description: 'User last name', example: 'Doe', type: 'string' },
      avatar: { description: 'User avatar URL', example: 'https://example.com/avatar.jpg', type: 'string', nullable: true },
      level: { description: 'User level', example: 5, type: 'number' },
      xp: { description: 'User XP', example: 1500, type: 'number' },
      streak: { description: 'User streak', example: 7, type: 'number' },
      teamId: { description: 'Team ID', example: 'team-123', type: 'string', nullable: true },
    },
  })
  user: {
    id: string;
    email: string;
    firstName: string;
    lastName: string;
    avatar?: string;
    level: number;
    xp: number;
    streak: number;
    teamId?: string;
  };
}