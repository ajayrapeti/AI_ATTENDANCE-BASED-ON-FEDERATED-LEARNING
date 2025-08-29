import { Controller, Get, Post, Body, Query, UseGuards, Param } from '@nestjs/common';
import { MarksService } from './marks.service';
import { CreateMarkDto } from './dto/create-mark.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ApiTags, ApiOperation, ApiResponse, ApiQuery, ApiBearerAuth } from '@nestjs/swagger';

@ApiTags('Marks')
@ApiBearerAuth()
@Controller('marks')
export class MarksController {
  constructor(private readonly marksService: MarksService) {}

  @Post()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Create or update a mark' })
  @ApiResponse({
    status: 201,
    description: 'Mark created/updated successfully',
  })
  async create(@Body() createMarkDto: CreateMarkDto) {
    return this.marksService.create(createMarkDto);
  }

  @Get('student/:id')
  @ApiQuery({ name: 'subjectName', required: false, type: String })
  @ApiResponse({ status: 200, description: 'Returns student marks' })
  @ApiResponse({ status: 404, description: 'Student not found' })
  getStudentMarks(
    @Param('id') id: string,
    @Query('subjectName') subjectName?: string
  ) {
    return this.marksService.getStudentMarks(id, subjectName);
  }

  @Get('percentage')
  @UseGuards(JwtAuthGuard)
  @ApiOperation({
    summary: 'Get marks percentages for all students with optional threshold filter',
  })
  @ApiQuery({
    name: 'threshold',
    required: false,
    type: Number,
    description: 'Minimum marks threshold',
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
    enum: ['mid1', 'mid2', 'assignment1', 'assignment2', 'quiz', 'attendance', 'weightedMid', 'total'],
    description: 'Filter by exam type or total marks',
  })
  async getMarksPercentages(
    @Query('threshold') threshold?: number,
    @Query('filter') filter?: 'above' | 'below',
    @Query('examType') examType?: 'mid1' | 'mid2' | 'assignment1' | 'assignment2' | 'quiz' | 'attendance' | 'weightedMid' | 'total',
  ) {
    return this.marksService.getMarksPercentages(threshold, filter, examType);
  }

  @Get('total')
  @ApiQuery({ name: 'threshold', required: false, type: Number })
  @ApiQuery({ name: 'filter', required: false, enum: ['above', 'below'] })
  @ApiQuery({ name: 'examType', required: false, enum: ['mid1', 'mid2', 'assignment1', 'assignment2', 'quiz', 'attendance', 'total'] })
  @ApiQuery({ name: 'subjectName', required: false, type: String })
  @ApiResponse({ status: 200, description: 'Returns total marks statistics' })
  async getTotalMarks(
    @Query('threshold') threshold?: number,
    @Query('filter') filter?: 'above' | 'below',
    @Query('examType') examType?: 'mid1' | 'mid2' | 'assignment1' | 'assignment2' | 'quiz' | 'attendance' | 'total',
    @Query('subjectName') subjectName?: string
  ) {
    console.log("Received request with params:", {
      threshold,
      filter,
      examType,
      subjectName
    });
    return this.marksService.getTotalMarks(threshold, filter, examType, subjectName);
  }
} 