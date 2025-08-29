import { Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn } from 'typeorm';
import { Student } from './student.entity';
import { Subject } from './subject.entity';

@Entity()
export class Attendance {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'date' })
  date: Date;

  @Column()
  period: number;

  @Column()
  present: boolean;

  @ManyToOne(() => Student, student => student.attendances)
  @JoinColumn()
  student: Student;

  @ManyToOne(() => Subject, subject => subject.attendances)
  @JoinColumn()
  subject: Subject;
} 