import { IsNotEmpty, IsString, IsNumber, IsBoolean, IsArray } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

class PeriodAttendanceDto {
  @ApiProperty()
  @IsNumber()
  period: number;

  @ApiProperty()
  @IsBoolean()
  present: boolean;
}

export class CreateAttendanceDto {
  @ApiProperty()
  @IsString()
  studentId: string;

  @ApiProperty()
  @IsString()
  subjectName: string;

  @ApiProperty({ type: [PeriodAttendanceDto] })
  @IsArray()
  periods: PeriodAttendanceDto[];
} 