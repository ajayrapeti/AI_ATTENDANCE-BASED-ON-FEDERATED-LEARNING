import { Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn } from 'typeorm';
import { Student } from './student.entity';
import { Subject } from './subject.entity';

@Entity()
export class Mark {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  examType: string; // 'mid1', 'mid2', 'assignment1', etc.

  @Column('decimal', { precision: 5, scale: 2 })
  score: number;

  @ManyToOne(() => Student, student => student.marks)
  @JoinColumn()
  student: Student;

  @ManyToOne(() => Subject, subject => subject.marks)
  @JoinColumn()
  subject: Subject;
} 