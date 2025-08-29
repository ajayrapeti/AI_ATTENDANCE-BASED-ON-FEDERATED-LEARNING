import { Controller, Get, Post, Body, Param, UseGuards } from '@nestjs/common';
import { EventsService } from './events.service';
import { CreateEventDto } from './dto/create-event.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';

@ApiTags('Events')
// @UseGuards(JwtAuthGuard)
@Controller('events')
export class EventsController {
  constructor(private readonly eventsService: EventsService) {}

  @Post()
  @ApiOperation({ summary: 'Create a new event' })
  @ApiResponse({
    status: 201,
    description: 'Event created successfully',
  })
  async create(@Body() createEventDto: CreateEventDto) {
    return this.eventsService.create(createEventDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all upcoming events' })
  @ApiResponse({
    status: 200,
    description: 'Retrieved all upcoming events successfully',
  })
  async findAll() {
    return this.eventsService.findAll();
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get an event by id' })
  @ApiResponse({
    status: 200,
    description: 'Retrieved event successfully',
  })
  async findOne(@Param('id') id: string) {
    return this.eventsService.findOne(id);
  }
}

