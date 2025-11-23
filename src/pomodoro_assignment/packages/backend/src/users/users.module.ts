import { forwardRef, Module } from '@nestjs/common';
import { UsersController } from './users.controller';
import { UsersService } from './users.service';
import { CoreModule } from '@/core.module';
 
@Module({
  imports: [
        forwardRef(() => CoreModule),
    
   ],
  controllers: [UsersController],
  providers: [UsersService],
  exports: [UsersService],
})
export class UsersModule {}