import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Between, Repository } from 'typeorm';
import { Attendance } from '../entities/attendance.entity';
import { Student } from '../entities/student.entity';
import { Subject } from '../entities/subject.entity';
import { CreateAttendanceDto } from './dto/create-attendance.dto';

@Injectable()
export class AttendanceService {
  private readonly TOTAL_SEMESTER_CLASSES = 90; // Hardcoded total classes per semester

  constructor(
    @InjectRepository(Attendance)
    private attendanceRepository: Repository<Attendance>,
    @InjectRepository(Student)
    private studentRepository: Repository<Student>,
    @InjectRepository(Subject)
    private subjectRepository: Repository<Subject>,
  ) {}

  private getISTDate(): Date {
    const now = new Date();
    const istTime = new Date(now.getTime() + 5.5 * 60 * 60 * 1000);
    istTime.setUTCHours(0, 0, 0, 0); // Set to UTC midnight for consistent date storage
    return istTime;
  }

  async createAttendance(createAttendanceDto: CreateAttendanceDto) {
    const { studentId, periods, subjectName } = createAttendanceDto;
    const today = this.getISTDate();

    const student = await this.studentRepository.findOne({
      where: { rollNo: studentId },
    });

    if (!student) {
      throw new NotFoundException(`Student with roll number ${studentId} not found`);
    }

    // Find subject by name
    const subject = await this.subjectRepository.findOne({
      where: { name: subjectName }
    });

    if (!subject) {
      throw new NotFoundException(`Subject ${subjectName} not found`);
    }

    // Process each period with the found subject
    const attendanceRecords = await Promise.all(
      periods.map(async (period) => {
        const existingRecord = await this.attendanceRepository.findOne({
          where: {
            student: { id: student.id },
            date: today,
            period: period.period,
          },
        });

        if (existingRecord) {
          existingRecord.present = period.present;
          existingRecord.subject = subject;
          return this.attendanceRepository.save(existingRecord);
        } else {
          const attendance = this.attendanceRepository.create({
            student,
            subject,
            date: today,
            period: period.period,
            present: period.present,
          });
          return this.attendanceRepository.save(attendance);
        }
      }),
    );

    return attendanceRecords;
  }

  private formatDate(date: Date | string): string {
    if (typeof date === 'string') {
      return date.split('T')[0];
    }
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }

  private convertToIST(date: string | Date): Date {
    const d = new Date(date);
    d.setHours(5, 30, 0, 0);
    return d;
  }

  async getAttendance(studentId: string, date?: string) {
    const queryDate = this.convertToIST(date || new Date());
    const formattedDate = this.formatDate(queryDate);

    const startDate = new Date(`${formattedDate}T00:00:00+05:30`);
    const endDate = new Date(`${formattedDate}T23:59:59+05:30`);

    const student = await this.studentRepository.findOne({
      where: { rollNo: studentId },
    });

    if (!student) {
      throw new NotFoundException(
        `Student with roll number ${studentId} not found`,
      );
    }

    const attendanceRecords = await this.attendanceRepository.find({
      where: {
        student: { id: student.id },
        date: Between(startDate, endDate),
      },
      relations: ['subject'],
      order: {
        period: 'ASC',
      },
    });

    return attendanceRecords.map((record) => ({
      ...record,
      date: this.formatDate(record.date),
    }));
  }

  async getAttendancePercentages(
    threshold?: number,
    filter?: 'above' | 'below',
    minThreshold?: number,
    maxThreshold?: number
  ) {
    // Get all students
    const students = await this.studentRepository.find();

    // Calculate percentage for each student
    const attendanceStats = await Promise.all(
      students.map(async (student) => {
        const presentClasses = await this.attendanceRepository.count({
          where: {
            student: { id: student.id },
            present: true,
          },
        });

        const percentage = (presentClasses / this.TOTAL_SEMESTER_CLASSES) * 100;

        return {
          studentId: student.id,
          rollNumber: student.rollNo,
          name: student.name,
          totalSemesterClasses: this.TOTAL_SEMESTER_CLASSES,
          attendedClasses: presentClasses,
          attendancePercentage: Math.round(percentage * 100) / 100,
        };
      }),
    );

    // Apply filters if provided
    const filteredStats = attendanceStats.filter((stat) => {
      // Range filter
      if (minThreshold !== undefined && maxThreshold !== undefined) {
        return stat.attendancePercentage >= minThreshold && 
               stat.attendancePercentage <= maxThreshold;
      }
      // Single threshold filter
      if (threshold !== undefined && filter) {
        if (filter === 'above') {
          return stat.attendancePercentage >= threshold;
        } else if (filter === 'below') {
          return stat.attendancePercentage < threshold;
        }
      }
      return true;
    });

    return { data: filteredStats };
  }
}
