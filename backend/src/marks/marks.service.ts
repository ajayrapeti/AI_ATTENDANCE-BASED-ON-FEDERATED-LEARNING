import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Mark } from '../entities/mark.entity';
import { Student } from '../entities/student.entity';
import { Subject } from '../entities/subject.entity';
import { CreateMarkDto, MaxMarks } from './dto/create-mark.dto';

@Injectable()
export class MarksService {
  private readonly MAX_MARKS = {
    mid1: 20,
    mid2: 20,
    assignment1: 10,
    assignment2: 10,
    quiz: 10,
    attendance: 5
  };

  constructor(
    @InjectRepository(Mark)
    private markRepository: Repository<Mark>,
    @InjectRepository(Student)
    private studentRepository: Repository<Student>,
    @InjectRepository(Subject)
    private subjectRepository: Repository<Subject>,
  ) {}

  async create(createMarkDto: CreateMarkDto) {
    const { studentId, subjectName, examType, score } = createMarkDto;

    // Validate score against max marks
    const maxScore = MaxMarks[examType];
    if (score > maxScore) {
      throw new BadRequestException(
        `Score cannot exceed maximum marks (${maxScore}) for ${examType}`
      );
    }

    const student = await this.studentRepository.findOne({
      where: { rollNo: studentId },
    });

    if (!student) {
      throw new NotFoundException(`Student with roll number ${studentId} not found`);
    }

    // Find subject by name
    const subject = await this.subjectRepository.findOne({
      where: { name: subjectName },
    });

    if (!subject) {
      throw new NotFoundException(`Subject with name ${subjectName} not found`);
    }

    // Check if mark already exists
    const existingMark = await this.markRepository.findOne({
      where: {
        student: { id: student.id },
        subject: { id: subject.id },
        examType,
      },
    });

    if (existingMark) {
      // Update existing mark
      existingMark.score = score;
      return this.markRepository.save(existingMark);
    }

    // Create new mark
    const mark = this.markRepository.create({
      student,
      subject,
      examType,
      score,
    });

    return this.markRepository.save(mark);
  }

  async getStudentMarks(studentId: string, subjectName?: string) {
    const student = await this.studentRepository.findOne({
      where: { rollNo: studentId },
    });

    if (!student) {
      throw new NotFoundException(`Student with roll number ${studentId} not found`);
    }

    const whereClause: any = {
      student: { id: student.id },
    };

    if (subjectName) {
      whereClause.subject = { name: subjectName };
    }

    const marks = await this.markRepository.find({
      where: whereClause,
      relations: ['subject'],
      order: {
        subject: { name: 'ASC' },
      },
    });

    // Group marks by subject
    const groupedMarks = marks.reduce((acc, mark) => {
      const subjectId = mark.subject.id;
      if (!acc[subjectId]) {
        acc[subjectId] = {
          subjectId: mark.subject.id,
          subjectName: mark.subject.name,
          marks: {},
          total: 0
        };
      }
      acc[subjectId].marks[mark.examType] = mark.score;
      acc[subjectId].total += mark.score;
      return acc;
    }, {});

    return Object.values(groupedMarks);
  }

  async getMarksPercentages(
    threshold?: number,
    filter?: 'above' | 'below',
    examType?: string,
  ) {
    const students = await this.studentRepository.find();

    const marksStats = await Promise.all(
      students.map(async (student) => {
        const marks = await this.markRepository.find({
          where: { student: { id: student.id } }
        });

        // Group marks by exam type
        const examMarks = {
          mid1: 0,
          mid2: 0,
          assignment1: 0,
          assignment2: 0,
          quiz: 0,
          attendance: 0
        };

        marks.forEach(mark => {
          examMarks[mark.examType] = Number(mark.score);
        });

        // Calculate weighted mid marks
        const midMarks = [examMarks.mid1, examMarks.mid2];
        const maxMid = Math.max(...midMarks);
        const minMid = Math.min(...midMarks);
        const weightedMidMarks = (maxMid * 2/3) + (minMid * 1/3);

        // Calculate total marks
        const totalMarks = weightedMidMarks + 
                          examMarks.assignment1 + 
                          examMarks.assignment2 + 
                          examMarks.quiz + 
                          examMarks.attendance;

        return {
          studentId: student.id,
          rollNumber: student.rollNo,
          name: student.name,
          examMarks,
          weightedMidMarks: Math.round(weightedMidMarks * 100) / 100,
          totalMarks: Math.round(totalMarks * 100) / 100
        };
      })
    );

    // Apply threshold filter if provided
    if (threshold !== undefined && filter) {
      const thresholdValue = Number(threshold);
      return {
        data: marksStats.filter((stat) => {
          if (filter === 'above') {
            return stat.totalMarks >= thresholdValue;
          } else if (filter === 'below') {
            return stat.totalMarks < thresholdValue;
          }
          return true;
        }),
      };
    }

    return { data: marksStats };
  }

  async getTotalMarks(
    threshold?: number,
    filter?: 'above' | 'below',
    examType?: 'mid1' | 'mid2' | 'assignment1' | 'assignment2' | 'quiz' | 'attendance' | 'total',
    subjectName?: string
  ) {
    const students = await this.studentRepository.find();

    const marksStats = await Promise.all(
      students.map(async (student) => {
        const whereClause: any = {
          student: { id: student.id },
        };

        if (subjectName && subjectName.trim() !== "") {
          whereClause.subject = { name: subjectName };
        }

        const marks = await this.markRepository.find({
          where: whereClause,
          relations: ['subject']
        });

        // Group marks by exam type
        const examMarks = {
          mid1: 0,
          mid2: 0,
          assignment1: 0,
          assignment2: 0,
          quiz: 0,
          attendance: 0
        };

        marks.forEach(mark => {
          examMarks[mark.examType] = Number(mark.score);
        });

        // Calculate weighted mid marks
        const midMarks = [examMarks.mid1, examMarks.mid2];
        const maxMid = Math.max(...midMarks);
        const minMid = Math.min(...midMarks);
        const weightedMidMarks = (maxMid * 2/3) + (minMid * 1/3);

        // Calculate total marks
        const totalMarks = weightedMidMarks + 
                          examMarks.assignment1 + 
                          examMarks.assignment2 + 
                          examMarks.quiz + 
                          examMarks.attendance;

        const result = {
          studentId: student.id,
          rollNumber: student.rollNo,
          name: student.name,
          examMarks,
          weightedMidMarks: Math.round(weightedMidMarks * 100) / 100,
          totalMarks: Math.round(totalMarks * 100) / 100
        };

        // Apply exam type filter if specified
        if (examType && examType !== 'total') {
          const markToCheck = examMarks[examType];
          const maxMark = this.MAX_MARKS[examType];
          
          if (threshold !== undefined && filter) {
            const thresholdValue = Number(threshold);
            if (filter === 'above') {
              return markToCheck >= thresholdValue ? result : null;
            } else {
              return markToCheck < thresholdValue ? result : null;
            }
          }
        } else if (examType === 'total' && threshold !== undefined && filter) {
          const thresholdValue = Number(threshold);
          if (filter === 'above') {
            return totalMarks >= thresholdValue ? result : null;
          } else {
            return totalMarks < thresholdValue ? result : null;
          }
        }

        return result;
      })
    );

    // Filter out null values and return results
    return {
      data: marksStats.filter(Boolean)
    };
  }
} 