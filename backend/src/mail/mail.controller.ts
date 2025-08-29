import { Controller, Post, Body, Request, Query } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiQuery, ApiBody } from '@nestjs/swagger';
import { Roles } from '../auth/decorators/roles.decorator';
import { Role } from '../auth/enums/role.enum';
import { MailService } from './mail.service';
import { SendEmailDto } from './dto/send-email.dto';
import { SendAttendanceWarningDto } from './dto/send-attendance-warning.dto';
import { SendAttendanceReportDto } from './dto/send-attendance-report.dto';

@ApiTags('mail')
@Controller('mail')
@ApiBearerAuth()
export class MailController {
  constructor(private readonly mailService: MailService) {}

  @Post('send')
  @Roles(Role.Admin, Role.Proctor, Role.Teacher)
  @ApiResponse({ status: 401, description: 'Unauthorized' })
  @ApiResponse({ status: 403, description: 'Forbidden' })
  async sendEmail(@Request() req, @Body() emailData: SendEmailDto) {
    console.log('Mail Controller - User:', req.user);
    return this.mailService.sendEmail(emailData);
  }

  @Post('send-attendance-warning')
  @Roles(Role.Admin, Role.Proctor, Role.Teacher)
  @ApiOperation({ summary: 'Send attendance warning emails' })
  @ApiResponse({ status: 200, description: 'Warning emails sent successfully' })
  @ApiResponse({ status: 401, description: 'Unauthorized' })
  @ApiResponse({ status: 403, description: 'Forbidden' })
  async sendAttendanceWarning(@Request() req, @Body() data: SendAttendanceWarningDto) {
    console.log('Mail Controller - User:', req.user);
    return this.mailService.sendAttendanceWarning(data);
  }

  @Post('send-attendance-report')
  @Roles(Role.Admin, Role.Proctor, Role.Teacher)
  @ApiOperation({ summary: 'Send attendance report emails to students based on filters' })
  @ApiResponse({ status: 200, description: 'Attendance reports sent successfully' })
  @ApiQuery({
    name: 'threshold',
    required: false,
    type: Number,
    description: 'Single threshold value (0-100)',
  })
  @ApiQuery({
    name: 'filter',
    required: false,
    enum: ['above', 'below'],
    description: 'Filter above or below threshold',
  })
  @ApiQuery({
    name: 'minThreshold',
    required: false,
    type: Number,
    description: 'Minimum attendance percentage (0-100)',
  })
  @ApiQuery({
    name: 'maxThreshold',
    required: false,
    type: Number,
    description: 'Maximum attendance percentage (0-100)',
  })
  @ApiBody({
    type: SendAttendanceReportDto,
    required: false,
    description: 'Optional custom message for the email'
  })
  async sendAttendanceReport(
    @Query('threshold') threshold?: number,
    @Query('filter') filter?: 'above' | 'below',
    @Query('minThreshold') minThreshold?: number,
    @Query('maxThreshold') maxThreshold?: number,
    @Body() data?: SendAttendanceReportDto
  ) {
    return this.mailService.sendAttendanceReportToAll(
      threshold,
      filter,
      minThreshold,
      maxThreshold,
      data?.customMessage
    );
  }

  @Post('send-marks-report')
  @Roles(Role.Admin, Role.Proctor, Role.Teacher)
  @ApiOperation({ summary: 'Send marks report emails to students based on filters' })
  @ApiResponse({ status: 200, description: 'Marks reports sent successfully' })
  @ApiQuery({
    name: 'threshold',
    required: false,
    type: Number,
    description: 'Marks threshold value',
  })
  @ApiQuery({
    name: 'filter',
    required: false,
    enum: ['above', 'below'],
    description: 'Filter students above or below threshold',
  })
  @ApiQuery({
    name: 'examType',
    required: false,
    enum: ['mid1', 'mid2', 'assignment1', 'assignment2', 'quiz', 'attendance', 'total'],
    description: 'Filter by exam type',
  })
  @ApiBody({
    type: SendAttendanceReportDto,
    required: false,
    description: 'Optional custom message for the email'
  })
  async sendMarksReport(
    @Query('threshold') threshold?: number,
    @Query('filter') filter?: 'above' | 'below',
    @Query('examType') examType?: 'mid1' | 'mid2' | 'assignment1' | 'assignment2' | 'quiz' | 'attendance' | 'total',
    @Body() data?: SendAttendanceReportDto
  ) {
    return this.mailService.sendMarksReportToAll(
      threshold,
      filter,
      examType,
      data?.customMessage
    );
  }
}
