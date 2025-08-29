import { IsArray, IsString, IsNotEmpty } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class SendAttendanceWarningDto {
  @ApiProperty({
    description: 'Array of email addresses to send the warning to',
    example: ['student@example.com']
  })
  @IsArray()
  @IsNotEmpty()
  to: string[];

  @ApiProperty({
    description: 'Warning message content',
    example: 'Your attendance has fallen below the required threshold.'
  })
  @IsString()
  @IsNotEmpty()
  message: string;
} 