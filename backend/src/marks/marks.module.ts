import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { MarksController } from './marks.controller';
import { MarksService } from './marks.service';
import { Mark } from '../entities/mark.entity';
import { Student } from '../entities/student.entity';
import { Subject } from '../entities/subject.entity';

@Module({
  imports: [TypeOrmModule.forFeature([Mark, Student, Subject])],
  controllers: [MarksController],
  providers: [MarksService],
  exports: [MarksService],
})
export class MarksModule {} 