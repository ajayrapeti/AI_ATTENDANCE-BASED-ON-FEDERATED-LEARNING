import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as SendGrid from '@sendgrid/mail';
import { SendEmailDto } from './dto/send-email.dto';
import { SendAttendanceWarningDto } from './dto/send-attendance-warning.dto';
import { AttendanceService } from '../attendance/attendance.service';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Student } from '../entities/student.entity';
import { MarksService } from '../marks/marks.service';

@Injectable()
export class MailService {
  constructor(
    private readonly configService: ConfigService,
    private readonly attendanceService: AttendanceService,
    @InjectRepository(Student)
    private studentRepository: Repository<Student>,
    private readonly marksService: MarksService
  ) {
    SendGrid.setApiKey(this.configService.get<string>('SENDGRID_API_KEY'));
  }

  async sendEmail(emailData: SendEmailDto) {
    try {
      const msg = {
        to: emailData.to,
        from: this.configService.get<string>('SENDGRID_VERIFIED_SENDER'),
        subject: emailData.subject,
        text: emailData.text,
        html: emailData.html,
      };

      console.log('Sending email with data:', msg);
      await SendGrid.send(msg);

      return {
        code: 200,
        msg: 'Email sent successfully',
        data: null,
      };
    } catch (error) {
      console.error('SendGrid Error:', error);
      if (error.response) {
        console.error('Error body:', error.response.body);
      }
      throw new InternalServerErrorException(error.message);
    }
  }

  async sendAttendanceWarning(data: SendAttendanceWarningDto) {
    try {
      const msg = {
        to: data.to,
        from: this.configService.get<string>('SENDGRID_VERIFIED_SENDER'),
        subject: 'Attendance Warning',
        text: data.message,
        html: `<div><h1>Attendance Warning</h1><p>${data.message}</p></div>`,
      };

      console.log('Sending attendance warning with data:', msg);
      await SendGrid.send(msg);

      return {
        code: 200,
        msg: 'Warning email sent successfully',
        data: null,
      };
    } catch (error) {
      console.error('SendGrid Error:', error);
      if (error.response) {
        console.error('Error body:', error.response.body);
      }
      throw new InternalServerErrorException(error.message);
    }
  }

  async sendAttendanceReportToAll(
    threshold?: number,
    filter?: 'above' | 'below',
    minThreshold?: number,
    maxThreshold?: number,
    customMessage?: string
  ) {
    // Get all attendance stats
    const { data: allStats } = await this.attendanceService.getAttendancePercentages();
    
    // Apply filters
    const attendanceStats = allStats.filter(stat => {
      // Range filter
      if (minThreshold !== undefined && maxThreshold !== undefined) {
        return stat.attendancePercentage >= minThreshold && 
               stat.attendancePercentage <= maxThreshold;
      }
      // Single threshold filter
      if (threshold !== undefined && filter) {
        if (filter === 'above') {
          return stat.attendancePercentage >= threshold;
        } else {
          return stat.attendancePercentage < threshold;
        }
      }
      return true;
    });

    // Rest of the email sending logic remains the same
    const students = await this.studentRepository.find();
    
    const results = await Promise.all(
      students.map(async (student) => {
        const studentStats = attendanceStats.find(
          (stat) => stat.rollNumber === student.rollNo
        );

        if (!studentStats || !student.email) {
          return { 
            student: student.rollNo, 
            status: 'skipped',
            reason: !studentStats ? 'Not in percentage range' : 'No email address found'
          };
        }

        const filterMessage = threshold && filter
          ? `\nYour attendance is ${filter} ${threshold}%`
          : minThreshold !== undefined && maxThreshold !== undefined
          ? `\nYour attendance falls between ${minThreshold}% and ${maxThreshold}%`
          : '';

        const rangeMessage = minThreshold !== undefined && maxThreshold !== undefined
          ? `\nYour attendance falls between ${minThreshold}% and ${maxThreshold}%`
          : '';

        const emailContent = `
          Dear ${student.name},

          Here is your current attendance report:
          Total Classes: ${studentStats.totalSemesterClasses}
          Classes Attended: ${studentStats.attendedClasses}
          Attendance Percentage: ${studentStats.attendancePercentage}%
          ${rangeMessage}
          ${filterMessage}
          ${customMessage ? `\n\nNote: ${customMessage}` : ''}

          Best regards,
          College Administration
        `;

        try {
          await this.sendEmail({
            to: [student.email],
            subject: 'Attendance Report',
            text: emailContent,
            html: `<div>${emailContent.split('\n').join('<br>')}</div>`
          });
          return { 
            student: student.rollNo, 
            status: 'success',
            percentage: studentStats.attendancePercentage 
          };
        } catch (error) {
          return { 
            student: student.rollNo, 
            status: 'failed',
            error: error.message 
          };
        }
      })
    );

    // Count successful emails
    const successfulEmails = results.filter(result => result.status === 'success').length;

    return {
      message: threshold && filter
        ? `Attendance reports sent to students ${filter} ${threshold}%`
        : minThreshold !== undefined && maxThreshold !== undefined
        ? `Attendance reports sent to students between ${minThreshold}% and ${maxThreshold}%`
        : 'Attendance reports sent to all students',
      results,
      totalSelected: attendanceStats.length,
      emailsSent: successfulEmails,
      totalStudents: students.length
    };
  }

  async sendMarksReportToAll(
    threshold?: number,
    filter?: 'above' | 'below',
    examType?: 'mid1' | 'mid2' | 'assignment1' | 'assignment2' | 'quiz' | 'attendance' | 'total',
    customMessage?: string
  ) {
    const { data: marksStats } = await this.marksService.getTotalMarks(threshold, filter, examType);
    
    const students = await this.studentRepository.find();
    
    const results = await Promise.all(
      students.map(async (student) => {
        const studentStats = marksStats.find(
          (stat) => stat.rollNumber === student.rollNo
        );

        if (!studentStats || !student.email) {
          return { 
            student: student.rollNo, 
            status: 'skipped',
            reason: !studentStats ? `Not in filter criteria` : 'No email address found'
          };
        }

        const examTypeMessage = examType 
          ? `\nExam: ${examType.toUpperCase()}`
          : '';

        const filterMessage = threshold && filter 
          ? `\nYour marks are ${filter} ${threshold}`
          : '';

        const marksDetails = examType && examType !== 'total'
          ? `\nMarks in ${examType}: ${studentStats.examMarks[examType]}`
          : `\nTotal Marks: ${studentStats.totalMarks}
             Mid1: ${studentStats.examMarks.mid1}
             Mid2: ${studentStats.examMarks.mid2}
             Assignment1: ${studentStats.examMarks.assignment1}
             Assignment2: ${studentStats.examMarks.assignment2}
             Quiz: ${studentStats.examMarks.quiz}
             Attendance: ${studentStats.examMarks.attendance}
             Weighted Mid Marks: ${studentStats.weightedMidMarks}`;

        const emailContent = `
          Dear ${student.name},

          Here is your marks report:
          ${examTypeMessage}
          ${marksDetails}
          ${filterMessage}
          ${customMessage ? `\n\nNote: ${customMessage}` : ''}

          Best regards,
          College Administration
        `;

        try {
          await this.sendEmail({
            to: [student.email],
            subject: 'Academic Performance Report',
            text: emailContent,
            html: `<div>${emailContent.split('\n').join('<br>')}</div>`
          });
          return { 
            student: student.rollNo, 
            status: 'success',
            marks: examType ? studentStats.examMarks[examType] : studentStats.totalMarks
          };
        } catch (error) {
          return { 
            student: student.rollNo, 
            status: 'failed',
            error: error.message 
          };
        }
      })
    );

    return {
      message: `Marks reports sent to students${threshold ? ` with ${filter} ${threshold} marks` : ''}${examType ? ` for ${examType}` : ''}`,
      results
    };
  }
} 