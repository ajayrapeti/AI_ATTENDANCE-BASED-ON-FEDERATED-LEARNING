import { Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn } from 'typeorm';
import { Student } from './student.entity';

@Entity()
export class Remark {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'date' })
  date: Date;

  @Column()
  teacher: string;

  @Column('text')
  comment: string;

  @ManyToOne(() => Student, student => student.remarks)
  @JoinColumn()
  student: Student;
} 