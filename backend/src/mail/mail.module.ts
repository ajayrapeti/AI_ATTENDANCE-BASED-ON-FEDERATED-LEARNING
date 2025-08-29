import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { MailController } from './mail.controller';
import { MailService } from './mail.service';
import { Student } from '../entities/student.entity';
import { AttendanceModule } from '../attendance/attendance.module';
import { MarksModule } from '../marks/marks.module';
import { ConfigModule } from '@nestjs/config';

@Module({
  imports: [
    TypeOrmModule.forFeature([Student]),
    AttendanceModule,
    MarksModule,
    ConfigModule,
  ],
  controllers: [MailController],
  providers: [MailService],
  exports: [MailService],
})
export class MailModule {} 