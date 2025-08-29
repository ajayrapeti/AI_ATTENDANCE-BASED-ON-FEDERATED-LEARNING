import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Event } from '../entities/event.entity';
import { CreateEventDto } from './dto/create-event.dto';

@Injectable()
export class EventsService {
  constructor(
    @InjectRepository(Event)
    private eventRepository: Repository<Event>,
  ) {}

  private formatDate(date: Date | string): string {
    if (typeof date === 'string') {
      return date.split('T')[0];
    }
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }

  async create(createEventDto: CreateEventDto) {
    const event = this.eventRepository.create({
      ...createEventDto,
      date: new Date(createEventDto.date),
    });
    
    const savedEvent = await this.eventRepository.save(event);
    return {
      ...savedEvent,
      date: this.formatDate(savedEvent.date)
    };
  }

  async findAll() {
    const events = await this.eventRepository.find({
      order: {
        date: 'DESC',
      },
    });

    return events.map(event => ({
      ...event,
      date: this.formatDate(event.date)
    }));
  }

  async findOne(id: string) {
    const event = await this.eventRepository.findOne({
      where: { id },
    });

    if (!event) {
      throw new NotFoundException(`Event with ID ${id} not found`);
    }

    return {
      ...event,
      date: this.formatDate(event.date)
    };
  }
} 