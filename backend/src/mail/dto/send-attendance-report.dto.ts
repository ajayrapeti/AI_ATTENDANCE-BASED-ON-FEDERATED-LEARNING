import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsOptional } from 'class-validator';

export class SendAttendanceReportDto {
  @ApiProperty({
    required: false,
    description: 'Custom message to include in the email (optional)'
  })
  @IsOptional()
  @IsString()
  customMessage?: string;
} 