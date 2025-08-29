import { IsArray, IsString, IsOptional, IsEmail, IsNotEmpty } from 'class-validator';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

export class SendEmailDto {
  @ApiProperty({ 
    description: 'Array of recipient email addresses',
    example: ['student1@example.com', 'student2@example.com']
  })
  @IsArray()
  @IsEmail({}, { each: true })
  to: string[];

  @ApiProperty({ 
    description: 'Email subject',
    example: 'Important Notice'
  })
  @IsString()
  @IsNotEmpty()
  subject: string;

  @ApiProperty({ 
    description: 'Email body in plain text',
    example: 'This is the email content'
  })
  @IsString()
  @IsNotEmpty()
  text: string;

  @ApiPropertyOptional({ 
    description: 'Email body in HTML format',
    example: '<h1>This is the email content</h1>'
  })
  @IsString()
  @IsOptional()
  html?: string;
} 