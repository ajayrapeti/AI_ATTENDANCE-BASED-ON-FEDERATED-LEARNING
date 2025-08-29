import { IsNotEmpty, IsString } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class LoginDto {
  @ApiProperty({
    example: 'A21126551057',
    description: 'Student roll number'
  })
  @IsNotEmpty()
  @IsString()
  rollNumber: string;

  @ApiProperty({
    example: 'password123',
    description: 'Student password'
  })
  @IsNotEmpty()
  @IsString()
  password: string;
} 