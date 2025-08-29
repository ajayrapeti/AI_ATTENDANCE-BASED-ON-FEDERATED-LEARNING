import { Entity, PrimaryGeneratedColumn, Column, OneToMany } from 'typeorm';
import { Attendance } from './attendance.entity';
import { Mark } from './mark.entity';
;

@Entity()
export class Subject {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  name: string;

  @OneToMany(() => Attendance, attendance => attendance.subject)
  attendances: Attendance[];

  @OneToMany(() => Mark, mark => mark.subject)
  marks: Mark[];
} 