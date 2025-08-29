import { IsNotEmpty, IsString } from 'class-validator';
import { Transform } from 'class-transformer';
import { ApiProperty } from '@nestjs/swagger';

export class CreateSubjectDto {
  @ApiProperty({
    example: 'Mathematics',
    description: 'Name of the subject'
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
  name: string;
} 