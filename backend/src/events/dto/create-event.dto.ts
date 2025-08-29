import { IsNotEmpty, IsString, IsDateString, IsEnum } from 'class-validator';
import { Transform } from 'class-transformer';
import { ApiProperty } from '@nestjs/swagger';

export enum EventType {
  HOLIDAY = 'holiday',
  EXAM = 'exam',
  ACTIVITY = 'activity'
}

export class CreateEventDto {
  @ApiProperty({
    example: 'Diwali Holiday',
    description: 'Title of the event'
  })
  @IsNotEmpty()
  @IsString()
  title: string;

  @ApiProperty({
    example: '2024-12-30',
    description: 'Date of the event'
  })
  @IsDateString()
  date: string;

  @ApiProperty({
    example: 'exam',
    description: 'Type of event (holiday, exam, activity)',
    enum: EventType
  })
  @IsEnum(EventType)
  @Transform(({ value }) => value.toLowerCase())
  type: EventType;
} 