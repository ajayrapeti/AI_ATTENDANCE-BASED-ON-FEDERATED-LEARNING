import { Injectable, UnauthorizedException, ConflictException, NotFoundException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { User, UserRole } from '../entities/user.entity';
import { Student } from '../entities/student.entity';
import { RegisterDto } from './dto/register.dto';
import { LoginDto } from './dto/login.dto';
import * as bcrypt from 'bcrypt';
import { AttendanceService } from '../attendance/attendance.service';
import { MarksService } from '../marks/marks.service';
import { EventsService } from '../events/events.service';

@Injectable()
export class AuthService {
  constructor(
    @InjectRepository(User)
    private userRepository: Repository<User>,
    @InjectRepository(Student)
    private studentRepository: Repository<Student>,
    private jwtService: JwtService,
    private attendanceService: AttendanceService,
    private marksService: MarksService,
    private eventsService: EventsService,
  ) {}

  async register(registerDto: RegisterDto) {
    // Check if user exists
    const existingUser = await this.userRepository.findOne({
      where: { rollNo: registerDto.rollNumber }
    });

    if (existingUser) {
      throw new ConflictException('User with this roll number already exists');
    }

    // Determine user role
    let role = UserRole.STUDENT;
    if (registerDto.isTeacher) {
      role = UserRole.TEACHER;
    } else if (registerDto.isProctor) {
      role = UserRole.PROCTOR;
    }

    // Create user with role and subject if teacher
    const user = this.userRepository.create({
      rollNo: registerDto.rollNumber,
      firstName: registerDto.firstName,
      lastName: registerDto.lastName,
      password: registerDto.password,
      role: role,
      subject: registerDto.isTeacher ? registerDto.subject : null
    });

    // Save user
    const savedUser = await this.userRepository.save(user);

    // If it's a student, create student record
    if (role === UserRole.STUDENT) {
      const student = this.studentRepository.create({
        rollNo: registerDto.rollNumber,
        name: `${registerDto.firstName} ${registerDto.lastName}`,
        class: registerDto.class,
        email: registerDto.email
      });
      await this.studentRepository.save(student);
    }
    
    const { password, ...result } = savedUser;
    return result;
  }

  async login(loginDto: LoginDto) {
    const user = await this.userRepository.findOne({
      where: { rollNo: loginDto.rollNumber },
      select: ['id', 'rollNo', 'firstName', 'lastName', 'password', 'role', 'subject'],
    });

    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const isPasswordValid = await bcrypt.compare(loginDto.password, user.password);
    if (!isPasswordValid) {
      throw new UnauthorizedException('Invalid credentials');
    }

    // Create JWT payload with all necessary user information
    const payload = { 
      sub: user.id, 
      rollNumber: user.rollNo,
      role: user.role,    // Make sure role is included
      firstName: user.firstName,
      lastName: user.lastName,
      subject: user.subject
    };

    console.log('Login - JWT Payload:', payload); // Debug log

    return {
      access_token: await this.jwtService.signAsync(payload),
      user: {
        id: user.id,
        rollNumber: user.rollNo,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role,
        subject: user.subject
      },
    };
  }

  async getProfileDetails(rollNumber: string) {
    // Get student details
    const student = await this.studentRepository.findOne({
      where: { rollNo: rollNumber },
      select: ['id', 'name', 'rollNo', 'class']
    });

    if (!student) {
      throw new NotFoundException(`Student with roll number ${rollNumber} not found`);
    }

    // Get today's date for attendance
    const today = new Date().toISOString().split('T')[0];

    // Get all data in parallel
    const [marks, attendance, events] = await Promise.all([
      this.marksService.getStudentMarks(rollNumber),
      this.attendanceService.getAttendance(rollNumber, today),
      this.eventsService.findAll() // Get all upcoming events
    ]);

    // Return combined response
    return {
      student,
      marks,
      attendance,
      events
    };
  }

  async getAllStudents() {
    const students = await this.studentRepository.find({
      select: ['id', 'name', 'rollNo', 'class'],
      order: {
        rollNo: 'ASC'  // Order by roll number
      }
    });

    return students;
  }
} 