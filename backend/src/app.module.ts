import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Student } from './entities/student.entity';
import { Subject } from './entities/subject.entity';
import { Attendance } from './entities/attendance.entity';
import { Mark } from './entities/mark.entity';
import { Event } from './entities/event.entity';
import { Remark } from './entities/remark.entity';
import { User } from './entities/user.entity';
import { AuthModule } from './auth/auth.module';
import { AttendanceModule } from './attendance/attendance.module';
import { SubjectsModule } from './subjects/subjects.module';
import { LoggerModule } from './logger/logger.module';
import { EventsModule } from './events/events.module';
import { MarksModule } from './marks/marks.module';
import { APP_GUARD } from '@nestjs/core';
import { AuthRolesGuard } from './auth/guards/auth.guard';
import { MailModule } from './mail/mail.module';
import { TestModule } from './test/test.module';

@Module({
  imports: [
    ConfigModule.forRoot(),
    TypeOrmModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: (configService: ConfigService) => ({
        type: 'postgres',
        host: configService.get('DB_HOST'),
        port: +configService.get('DB_PORT'),
        username: configService.get('DB_USERNAME'),
        password: configService.get('DB_PASSWORD'),
        database: configService.get('DB_NAME'),
        entities: [Student, Subject, Attendance, Mark, Event, Remark, User],
        synchronize: true, // Set to false in production
        // Additional schema-related configurations
        migrations: ['dist/migrations/*.js'],
        migrationsTableName: 'migrations',
        migrationsRun: true,
      }),
      inject: [ConfigService],
    }),
    AuthModule,
    AttendanceModule,
    SubjectsModule,
    LoggerModule,
    EventsModule,
    MarksModule,
    MailModule,
    TestModule,
  ],
  providers: [
    {
      provide: APP_GUARD,
      useClass: AuthRolesGuard,
    },
  ],
})
export class AppModule {}