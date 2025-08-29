import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { JwtStrategy } from './strategies/jwt.strategy';
import { User } from '../entities/user.entity';
import { Student } from '../entities/student.entity';
import { AttendanceModule } from '../attendance/attendance.module';
import { MarksModule } from '../marks/marks.module';
import { EventsModule } from '../events/events.module';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { AuthRolesGuard } from './guards/auth.guard';

@Module({
  imports: [
    TypeOrmModule.forFeature([User, Student]),
    PassportModule.register({ defaultStrategy: 'jwt' }),
    JwtModule.registerAsync({
      imports: [ConfigModule],
      useFactory: async (configService: ConfigService) => ({
        secret: configService.get<string>('JWT_SECRET'),
        signOptions: { expiresIn: '24h' },
      }),
      inject: [ConfigService],
    }),
    AttendanceModule,
    MarksModule,
    EventsModule,
  ],
  controllers: [AuthController],
  providers: [
    AuthService,
    JwtStrategy,
    AuthRolesGuard
  ],
  exports: [
    AuthService,
    JwtModule,
    AuthRolesGuard
  ],
})
export class AuthModule {} 