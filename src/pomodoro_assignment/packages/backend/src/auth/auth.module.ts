import { Module, forwardRef } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
 import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { JwtStrategy } from './strategies/jwt.strategy';
import { CoreModule } from '../core.module';
 
@Module({
  imports: [
    forwardRef(() => CoreModule),
    PassportModule,
    JwtModule.register({
        secret: process.env.JWT_SECRET || 'default-secret-key',
        signOptions: {
          expiresIn: '1h',
        },
      }),
  ],
  controllers: [AuthController],
  providers: [
    AuthService,
    JwtStrategy,
  ],
  exports: [AuthService, JwtModule, PassportModule],
})
export class AuthModule {}