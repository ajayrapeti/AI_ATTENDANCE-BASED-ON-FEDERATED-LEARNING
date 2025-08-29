import { Entity, PrimaryGeneratedColumn, Column, OneToMany } from 'typeorm';
import { Attendance } from './attendance.entity';
import { Mark } from './mark.entity';
import { Remark } from './remark.entity';


@Entity()
export class Student {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  name: string;

  @Column()
  class: string;

  @Column()
  rollNo: string;

  @Column({ nullable: true })
  email: string;

  @OneToMany(() => Attendance, attendance => attendance.student)
  attendances: Attendance[];

  @OneToMany(() => Mark, mark => mark.student)
  marks: Mark[];

  @OneToMany(() => Remark, remark => remark.student)
  remarks: Remark[];
} 