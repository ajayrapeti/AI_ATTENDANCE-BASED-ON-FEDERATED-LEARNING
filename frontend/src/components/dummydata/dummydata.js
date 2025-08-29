// Dummy data object for Dashboard
const dummyData = {
  student: {
    name: "John Doe",
    class: "4/4 CSD",
    rollNo: "A21126551057",
  },
  dateWiseAttendance: {
    "2024-03-01": [
      { period: 1, subject: "Mathematics", present: true },
      { period: 2, subject: "Physics", present: true },
      { period: 3, subject: "Chemistry", present: false },
      { period: 4, subject: "English", present: true },
      { period: 5, subject: "Computer Science", present: true },
      { period: 6, subject: "History", present: true },
      { period: 7, subject: "Geography", present: true },
      { period: 8, subject: "Physical Education", present: true },
    ],
    "2024-03-04": [
      { period: 1, subject: "Physics", present: true },
      { period: 2, subject: "Mathematics", present: true },
      { period: 3, subject: "English", present: true },
      { period: 4, subject: "Computer Science", present: true },
      { period: 5, subject: "Chemistry", present: false },
      { period: 6, subject: "Physical Education", present: true },
      { period: 7, subject: "History", present: true },
      { period: 8, subject: "Geography", present: true },
    ],
    "2024-03-05": [
      { period: 1, subject: "Chemistry", present: true },
      { period: 2, subject: "English", present: true },
      { period: 3, subject: "Mathematics", present: true },
      { period: 4, subject: "History", present: true },
      { period: 5, subject: "Physics", present: true },
      { period: 6, subject: "Computer Science", present: true },
      { period: 7, subject: "Physical Education", present: true },
      { period: 8, subject: "Geography", present: false },
    ]
  },
  marks: {
    Chemistry: {
      mid1: 18,
      mid2: 19,
      assignment1: 8,
      assignment2: 9,
      quiz: 4,
      attendance: 5
    },
    English: {
      mid1: 17,
      mid2: 16,
      assignment1: 9,
      assignment2: 8,
      quiz: 4,
      attendance: 5
    },
    "Computer Science": {
      mid1: 19,
      mid2: 18,
      assignment1: 9,
      assignment2: 10,
      quiz: 5,
      attendance: 5
    },
    Mathematics: {
      mid1: 18,
      mid2: 17,
      assignment1: 8,
      assignment2: 9,
      quiz: 4,
      attendance: 4
    },
    Physics: {
      mid1: 16,
      mid2: 18,
      assignment1: 9,
      assignment2: 8,
      quiz: 4,
      attendance: 5
    }
  },
  events: {
    weeklyHolidays: ["Sunday", "Saturday"], // Add weekly holidays
    holidays: [
      { date: "2024-01-01", title: "New Year's Day" },
      { date: "2024-05-01", title: "Labor Day" },
      { date: "2024-08-15", title: "Republic Day" },
      { date: "2024-03-25", title: "Holi" },
      { date: "2024-04-17", title: "Ram Navami" },
    ],

    exams: [
      { date: "2024-12-10", title: "Final Exams Begin" },
      { date: "2024-12-15", title: "Mathematics Mid-term" },
      { date: "2024-12-16", title: "Physics Mid-term" },
      { date: "2024-12-18", title: "Chemistry Mid-term" },
      { date: "2024-12-20", title: "English Mid-term" },
    ],

    activities: [
      { date: "2024-10-05", title: "Cultural Fest" },
      { date: "2024-10-10", title: "Annual Sports Day" },
      { date: "2024-10-15", title: "Parent-Teacher Meeting" },
      { date: "2024-10-22", title: "Earth Day Celebration" },
      { date: "2024-10-28", title: "Science Exhibition" },
    ],
  },
  remarks: [
    {
      date: "2025-01-15",
      teacher: "Mrs. Smith",
      comment: "Excellent participation in class",
    },
    {
      date: "2025-01-10",
      teacher: "Mr. Johnson",
      comment: "Need to improve homework submission",
    },
    {
      date: "2025-01-05",
      teacher: "Ms. Davis",
      comment: "Outstanding performance in science project",
    },
    {
      date: "2025-01-28",
      teacher: "Mr. Wilson",
      comment: "Good improvement in mathematics",
    },
  ],
};

export default dummyData;
