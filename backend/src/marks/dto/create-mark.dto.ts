import { IsNotEmpty, IsString, IsNumber, IsEnum } from 'class-validator';
import { Transform } from 'class-transformer';
import { ApiProperty } from '@nestjs/swagger';

export enum ExamType {
  MID1 = 'mid1',
  MID2 = 'mid2',
  ASSIGNMENT1 = 'assignment1',
  ASSIGNMENT2 = 'assignment2',
  QUIZ = 'quiz',
  ATTENDANCE = 'attendance'
}

export const MaxMarks = {
  [ExamType.MID1]: 20,
  [ExamType.MID2]: 20,
  [ExamType.ASSIGNMENT1]: 10,
  [ExamType.ASSIGNMENT2]: 10,
  [ExamType.QUIZ]: 10,
  [ExamType.ATTENDANCE]: 5
};

export class CreateMarkDto {
  @ApiProperty({
    example: 'A21126551057',
    description: 'Student Roll Number'
  })
  @IsNotEmpty()
  @IsString()
  studentId: string;

  @ApiProperty({
    example: 'Mathematics',
    description: 'Subject Name'
  })
  @IsNotEmpty()
  @IsString()
  @Transform(({ value }) => {
    // Capitalize first letter of each word
    return value
      .toLowerCase()
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  })
  subjectName: string;

  @ApiProperty({
    example: 'mid1',
    description: 'Type of exam',
    enum: ExamType
  })
  @IsEnum(ExamType)
  @Transform(({ value }) => value.toLowerCase())
  examType: ExamType;

  @ApiProperty({
    example: 15,
    description: 'Score obtained'
  })
  @IsNumber()
  score: number;
} 