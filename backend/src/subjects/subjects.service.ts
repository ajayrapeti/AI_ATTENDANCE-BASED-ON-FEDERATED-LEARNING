import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Subject } from '../entities/subject.entity';
import { CreateSubjectDto } from './dto/create-subject.dto';

@Injectable()
export class SubjectsService {
  constructor(
    @InjectRepository(Subject)
    private subjectRepository: Repository<Subject>,
  ) {}

  async create(createSubjectDto: CreateSubjectDto) {
    const subject = this.subjectRepository.create(createSubjectDto);
    return await this.subjectRepository.save(subject);
  }

  async findAll() {
    const subjects = await this.subjectRepository.find({
      order: {
        name: 'ASC',
      },
    });

    if (!subjects.length) {
      throw new NotFoundException('No subjects found');
    }

    return subjects;
  }

  async findOne(id: string) {
    const subject = await this.subjectRepository.findOne({
      where: { id },
    });

    if (!subject) {
      throw new NotFoundException(`Subject with ID ${id} not found`);
    }

    return subject;
  }
} 