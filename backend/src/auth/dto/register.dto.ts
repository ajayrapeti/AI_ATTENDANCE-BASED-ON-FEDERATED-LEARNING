import { IsNotEmpty, IsString, IsBoolean, IsOptional } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class RegisterDto {
  @ApiProperty({
    example: 'A21126551057',
    description: 'Student/Teacher ID'
  })
  @IsNotEmpty()
  @IsString()
  rollNumber: string;

  @ApiProperty({
    example: 'John',
    description: 'First name'
  })
  @IsNotEmpty()
  @IsString()
  firstName: string;

  @ApiProperty({
    example: 'Doe',
    description: 'Last name'
  })
  @IsNotEmpty()
  @IsString()
  lastName: string;

  @ApiProperty({
    example: 'CSE-A',
    description: 'Class name (for students only)'
  })
  @IsOptional()
  @IsString()
  class?: string;

  @ApiProperty({
    example: false,
    description: 'Whether the user is a teacher'
  })
  @IsNotEmpty()
  @IsBoolean()
  isTeacher: boolean;

  @ApiProperty({
    example: 'React JS',
    description: 'Subject (for teachers only)'
  })
  @IsOptional()
  @IsString()
  subject?: string;

  @ApiProperty({
    example: 'password123',
    description: 'Password'
  })
  @IsNotEmpty()
  @IsString()
  password: string;

  @ApiProperty({
    example: false,
    description: 'Whether the user is a proctor'
  })
  @IsOptional()
  @IsBoolean()
  isProctor: boolean;

  @ApiProperty({
    example: 'john.doe@example.com',
    description: 'Email (required for students)'
  })
  @IsOptional()
  @IsString()
  email?: string;
} 