// Calendar.jsx
import React, { useState } from 'react';
import { format, isSameDay } from 'date-fns';
import { motion } from 'framer-motion';
import axios from 'axios';
import { useSession } from '../../contexts/SessionContext';

const Calendar = ({ attendance, events, studentId }) => {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedDayAttendance, setSelectedDayAttendance] = useState([]);
  const today = new Date();
  const { session } = useSession();

  // Calculate calendar values
  const daysInMonth = new Date(
    currentDate.getFullYear(),
    currentDate.getMonth() + 1,
    0
  ).getDate();

  const firstDayOfMonth = new Date(
    currentDate.getFullYear(),
    currentDate.getMonth(),
    1
  ).getDay();

  const renderCalendarDays = () => {
    const days = [];
    const totalDays = daysInMonth + firstDayOfMonth;
    const weeks = Math.ceil(totalDays / 7);

    for (let i = 0; i < weeks * 7; i++) {
      const dayNumber = i - firstDayOfMonth + 1;
      const isCurrentMonth = dayNumber > 0 && dayNumber <= daysInMonth;
      const date = isCurrentMonth 
        ? new Date(currentDate.getFullYear(), currentDate.getMonth(), dayNumber) 
        : null;

      days.push({
        number: isCurrentMonth ? dayNumber : null,
        date: date,
        isCurrentMonth: isCurrentMonth
      });
    }

    return days;
  };

  // Function to fetch attendance for the selected date
  const fetchAttendance = async (date) => {
    const formattedDate = format(date, 'yyyy-MM-dd');
    try {
      const token = session.token;
      const response = await axios.get(`http://localhost:3000/attendance?studentId=${studentId}&date=${formattedDate}`, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      setSelectedDayAttendance(response.data.data); // Assuming the response structure
      console.log(response)
    } catch (error) {
      console.error('Failed to fetch attendance:', error);
      alert('Failed to fetch attendance');
    }
  };

  const handleDateClick = (dayNumber) => {
    const date = new Date(currentDate.getFullYear(), currentDate.getMonth(), dayNumber);
    setSelectedDate(date);
    fetchAttendance(date); // Fetch attendance for the selected date
  };

  const getEventsForDate = (date) => {
    const dateStr = format(date, 'yyyy-MM-dd');
    const dayName = format(date, 'EEEE'); // Get day name
    
    return {
      holiday: events.holidays.find(h => h.date === dateStr) || 
              (events.weeklyHolidays.includes(dayName) ? { title: 'Weekly Holiday' } : null),
      exam: events.exams.find(e => e.date === dateStr),
      activity: events.activities.find(a => a.date === dateStr)
    };
  };

  return (
    <div>
      <div className="grid grid-cols-1 md:grid-cols-[2fr,1fr] gap-6 p-6">
        {/* Calendar section */}
        <div>
          {/* Calendar header */}
          <div className="flex justify-between items-center mb-6">
            <button 
              onClick={() => setCurrentDate(new Date(currentDate.setMonth(currentDate.getMonth() - 1)))}
              className="px-4 py-2 bg-secondary-700 hover:bg-secondary-600 text-white rounded-lg transition-colors"
            >
              Previous
            </button>
            <h2 className="text-xl font-semibold text-white">{format(currentDate, 'MMMM yyyy')}</h2>
            <button 
              onClick={() => setCurrentDate(new Date(currentDate.setMonth(currentDate.getMonth() + 1)))}
              className="px-4 py-2 bg-secondary-700 hover:bg-secondary-600 text-white rounded-lg transition-colors"
            >
              Next
            </button>
          </div>

          {/* Weekday headers */}
          <div className="grid grid-cols-7 gap-2 mb-2">
            {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
              <div key={day} className="text-center font-medium py-2 text-white/80">{day}</div>
            ))}
          </div>

          {/* Calendar grid */}
          <div className="grid grid-cols-7 gap-2">
            {renderCalendarDays().map((day, index) => (
              <div
                key={index}
                onClick={() => day.isCurrentMonth && handleDateClick(day.number)}
                className={`
                  relative aspect-square cursor-pointer rounded-lg
                  ${!day.isCurrentMonth ? 'bg-secondary-900/30 border border-gray-700 cursor-default' : 
                    selectedDate && day.date && isSameDay(day.date, selectedDate) ? 'border-2 border-primary-400 bg-primary-900/20' : 
                    day.date && isSameDay(day.date, today) ? 'border border-primary-400 bg-secondary-700/30' : 
                    day.isCurrentMonth ? 'border border-white/50 bg-secondary-800/30 hover:bg-secondary-700/30' :
                    'border border-gray-700 bg-secondary-800/30 hover:bg-secondary-700/30'}
                  transition-all duration-200 group
                `}
              >
                {day.isCurrentMonth && (
                  <>
                    {/* Date number */}
                    <span className={`
                      absolute top-1 left-1 z-10 text-sm
                      ${day.date && isSameDay(day.date, today) ? 'font-bold text-primary-400' : 
                        selectedDate && day.date && isSameDay(day.date, selectedDate) ? 'font-bold text-primary-300' : 
                        'text-white'}
                    `}>
                      {day.number}
                    </span>

                    {/* Event indicators */}
                    {day.date && getEventsForDate(day.date) && (
                      <div className="absolute bottom-1 left-1 right-1 flex gap-1">
                        {getEventsForDate(day.date).holiday && (
                          <div className="h-1.5 w-1.5 rounded-full bg-red-400" />
                        )}
                        {getEventsForDate(day.date).exam && (
                          <div className="h-1.5 w-1.5 rounded-full bg-yellow-400" />
                        )}
                        {getEventsForDate(day.date).activity && (
                          <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                        )}
                      </div>
                    )}

                    {/* Tooltip */}
                    {day.date && getEventsForDate(day.date) && (
                      Object.values(getEventsForDate(day.date)).some(event => event) && (
                        <div className="invisible group-hover:visible absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-secondary-800 text-white text-xs rounded whitespace-pre z-20 min-w-max border border-secondary-700">
                          {getEventsForDate(day.date).holiday && (
                            <div className="text-red-400">🔴 {getEventsForDate(day.date).holiday.title}</div>
                          )}
                          {getEventsForDate(day.date).exam && (
                            <div className="text-yellow-400">🟡 {getEventsForDate(day.date).exam.title}</div>
                          )}
                          {getEventsForDate(day.date).activity && (
                            <div className="text-green-400">🟢 {getEventsForDate(day.date).activity.title}</div>
                          )}
                          <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2 rotate-45 w-2 h-2 bg-secondary-800 border-r border-b border-secondary-700"></div>
                        </div>
                      )
                    )}
                  </>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Attendance details section */}
        <div className="h-full">
          {selectedDate ? (
            <div className="bg-secondary-900/50 rounded-xl p-6 border border-secondary-700/30">
              <h3 className="text-lg font-semibold text-white mb-4">
                {format(selectedDate, 'EEEE, MMMM dd, yyyy')}
              </h3>
              <div className="space-y-2">
                {selectedDayAttendance.length > 0 ? (
                  selectedDayAttendance.map((attendance) => (
                    <div 
                      key={attendance.id} 
                      className={`flex justify-between items-center p-3 rounded-lg ${
                        attendance.present ? 'bg-green-900/20' : 'bg-red-900/20'
                      }`}
                    >
                      <div className="flex flex-col">
                        <span className="font-medium text-white">
                          {attendance.subject.name}
                        </span>
                        <span className="text-sm text-white/60">
                          Period {attendance.period}
                        </span>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-sm ${
                        attendance.present 
                          ? 'bg-green-900/30 text-green-400' 
                          : 'bg-red-900/30 text-red-400'
                      }`}>
                        {attendance.present ? 'Present' : 'Absent'}
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-white/60">
                    No attendance data available
                  </div>
                )}
              </div>
            </div>
          ) : (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="h-full flex flex-col items-center justify-center p-6 bg-secondary-900/50 rounded-xl border border-secondary-700/30"
            >
              <p className="text-white/60 text-center">
                Select a date to view attendance details
              </p>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Calendar;