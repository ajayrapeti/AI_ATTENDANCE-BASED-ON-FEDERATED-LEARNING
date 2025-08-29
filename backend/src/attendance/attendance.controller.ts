import { Controller, Post, Get, Body, Query, UseGuards, Header } from '@nestjs/common';
import { AttendanceService } from './attendance.service';
import { CreateAttendanceDto } from './dto/create-attendance.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ApiTags, ApiOperation, ApiResponse, ApiQuery, ApiHeader, ApiBearerAuth, ApiBody } from '@nestjs/swagger';

@ApiTags('Attendance')
@ApiBearerAuth()
@Controller('attendance')
export class AttendanceController {
  constructor(private readonly attendanceService: AttendanceService) {}

  @Post()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({
    summary: 'Create attendance records for a student (for today)',
  })
  @ApiBody({
    description: 'Attendance record data',
    schema: {
      example: {
        studentId: "12345",
        subjectName: "Mathematics",
        periods: [
          {
            period: 1,
            present: true
          },
          {
            period: 2,
            present: false
          }
        ]
      }
    }
  })
  @ApiResponse({
    status: 201,
    description: 'Attendance records created successfully',
    schema: {
      example: {
        studentId: "12345",
        subjectName: "Mathematics",
        periods: [
          {
            period: 1,
            present: true
          },
          {
            period: 2,
            present: false
          }
        ]
      }
    }
  })
  
  async createAttendance(@Body() createAttendanceDto: CreateAttendanceDto) {
    return this.attendanceService.createAttendance(createAttendanceDto);
  }

  @Get()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({
    summary: 'Get attendance records for a student (defaults to today)',
  })
  @ApiResponse({
    status: 200,
    description: 'Attendance records retrieved successfully',
  })
  
  @ApiQuery({
    name: 'studentId',
    required: true,
    type: String,
    description: 'Student ID to fetch attendance for',
  })
  @ApiQuery({
    name: 'date',
    required: false,
    type: String,
    description: 'Date to fetch attendance for (YYYY-MM-DD)',
  })
  async getAttendance(
    @Query('studentId') studentId: string,
    @Query('date') date?: string,
  ) {
    return this.attendanceService.getAttendance(studentId, date);
  }

  @Get('percentage')
  @UseGuards(JwtAuthGuard)
  @ApiOperation({
    summary: 'Get attendance percentages for all students with filtering options',
  })
  @ApiResponse({
    status: 200,
    description: 'Returns attendance percentages for students',
    schema: {
      example: {
        data: [
          {
            studentId: '1',
            rollNumber: 'A123',
            name: 'John Doe',
            totalClasses: 50,
            presentClasses: 45,
            attendancePercentage: 90.00
          }
        ]
      }
    }
  })
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
  async getAttendancePercentages(
    @Query('threshold') threshold?: number,
    @Query('filter') filter?: 'above' | 'below',
    @Query('minThreshold') minThreshold?: number,
    @Query('maxThreshold') maxThreshold?: number,
  ) {
    return this.attendanceService.getAttendancePercentages(
      threshold,
      filter,
      minThreshold,
      maxThreshold
    );
  }
}
