import React, { useState, useEffect, useCallback } from "react";
import {
  Box,
  Container,
  AppBar,
  Tabs,
  Tab,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Button,
  Alert,
  Snackbar,
  Divider,
} from "@mui/material";
import { motion } from "framer-motion";
import { useSession } from "../contexts/SessionContext";

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Box>{children}</Box>
        </motion.div>
      )}
    </div>
  );
}

function Proctordashboard() {
  const [value, setValue] = useState(0);
  const [attendanceData, setAttendanceData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [threshold, setThreshold] = useState("");
  const [filter, setFilter] = useState("all");
  const [marksData, setMarksData] = useState([]);
  const [marksLoading, setMarksLoading] = useState(true);
  const [marksThreshold, setMarksThreshold] = useState("");
  const [marksFilter, setMarksFilter] = useState("all");
  const [examType, setExamType] = useState("mid1");
  const [customMessage, setCustomMessage] = useState("");
  const [emailStatus, setEmailStatus] = useState(null);
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const { session } = useSession();
  const [minThreshold, setMinThreshold] = useState("");
  const [maxThreshold, setMaxThreshold] = useState("");
  const [emailMinThreshold, setEmailMinThreshold] = useState("");
  const [emailMaxThreshold, setEmailMaxThreshold] = useState("");
  const [subject, setSubject] = useState("");
  const [subjectsList, setSubjectsList] = useState([]);

  useEffect(() => {
    const fetchSubjects = async () => {
      try {
        const response = await fetch('http://localhost:3000/subjects', {
          headers: {
            Authorization: `Bearer ${session.token}`,
            accept: "application/json",
          },
        });
        const result = await response.json();
        setSubjectsList(result.data);
      } catch (error) {
        console.error("Error fetching subjects:", error);
      }
    };

    if (session?.token) {
      fetchSubjects();
    }
  }, [session]);

  const fetchAttendanceData = useCallback(async () => {
    try {
      let url = "http://localhost:3000/attendance/percentage";

      const params = new URLSearchParams();
      if (threshold && filter !== "all") {
        params.append("threshold", threshold);
        params.append("filter", filter);
      }
      if (minThreshold && maxThreshold) {
        params.append("minThreshold", minThreshold);
        params.append("maxThreshold", maxThreshold);
      }

      if (params.toString()) {
        url += `?${params.toString()}`;
      }

      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${session.token}`,
          accept: "application/json",
        },
      });
      const result = await response.json();
      setAttendanceData(
        Array.isArray(result.data.data) ? result.data.data : []
      );
      setLoading(false);
    } catch (error) {
      console.error("Error fetching attendance data:", error);
      setAttendanceData([]);
      setLoading(false);
    }
  }, [session, filter, threshold, minThreshold, maxThreshold]);

  const fetchMarksData = useCallback(async () => {
    try {
      let url = "http://localhost:3000/marks/total";

      const params = new URLSearchParams();
      if (marksThreshold) params.append("threshold", marksThreshold);
      if (marksFilter !== "all") params.append("filter", marksFilter);
      if (examType) params.append("examType", examType);
      if (subject && subject.trim() !== "") {
        params.append("subjectName", subject);
      }

      if (params.toString()) {
        url += `?${params.toString()}`;
      }

      console.log("Sending request to:", url);
      console.log("Request params:", {
        marksThreshold,
        marksFilter,
        examType,
        subject,
        token: session.token
      });

      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${session.token}`,
          accept: "application/json",
        },
      });

      console.log("Response status:", response.status);
      const result = await response.json();
      console.log("Marks Data Response:", result);
      setMarksData(result.data.data || []);
      setMarksLoading(false);
    } catch (error) {
      console.error("Error fetching marks data:", error);
      setMarksData([]);
      setMarksLoading(false);
    }
  }, [session, marksThreshold, marksFilter, examType, subject]);

  useEffect(() => {
    if (session?.token) {
      fetchAttendanceData();
      fetchMarksData();
    }
  }, [session, fetchAttendanceData, fetchMarksData]);

  const handleChange = (event, newValue) => {
    setValue(newValue);
  };

  const handleCloseSnackbar = (event, reason) => {
    if (reason === "clickaway") {
      return;
    }
    setOpenSnackbar(false);
  };

  const handleSendEmails = async (threshold = null, filter = null, min = null, max = null) => {
    try {
      let url = 'http://localhost:3000/mail/send-attendance-report';
      
      const params = new URLSearchParams();
      if (threshold && filter) {
        params.append('threshold', threshold);
        params.append('filter', filter);
      }
      if (min && max) {
        params.append('minThreshold', min);
        params.append('maxThreshold', max);
      }
      
      if (params.toString()) {
        url += `?${params.toString()}`;
      }
  
      console.log('Sending request to:', url); // Debug log
  
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.token}`,
          'Content-Type': 'application/json'
        },
        body: customMessage ? JSON.stringify({ customMessage }) : null
      });
  
      const result = await response.json();
      console.log('Server response:', result); // Debug log
      
      if (response.ok) {
        setEmailStatus({ 
          severity: 'success',
          message: `${result.data.message} (${result.data.emailsSent}/${result.data.totalSelected} selected students out of ${result.data.totalStudents} total)`
        });
      } else {
        throw new Error(result.data?.message || result.msg || 'Failed to send emails');
      }
      setOpenSnackbar(true);
    } catch (error) {
      setEmailStatus({ 
        severity: 'error',
        message: error.message || 'Failed to send emails. Please try again.'
      });
      setOpenSnackbar(true);
      console.error('Error sending emails:', error);
    }
  };

  const handleSendMarksEmails = async (
    threshold = null,
    filter = null,
    examType = null
  ) => {
    try {
      let url = "http://localhost:3000/mail/send-marks-report";

      // Add query parameters if provided
      const params = new URLSearchParams();
      if (threshold) params.append("threshold", threshold);
      if (filter) params.append("filter", filter);
      if (examType) params.append("examType", examType);

      if (params.toString()) {
        url += `?${params.toString()}`;
      }

      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${session.token}`,
          "Content-Type": "application/json",
        },
        body: customMessage ? JSON.stringify({ customMessage }) : null,
      });

      const result = await response.json();
      setEmailStatus({ success: true, message: result.message });
      setOpenSnackbar(true);
    } catch (error) {
      setEmailStatus({ success: false, message: "Failed to send emails" });
      setOpenSnackbar(true);
      console.error("Error sending marks emails:", error);
    }
  };

  // Add handlers for filter changes
  const handleSingleFilterChange = (newThreshold, newFilter) => {
    // Reset range filter when using single filter
    setMinThreshold("");
    setMaxThreshold("");
    // Set single filter values
    setThreshold(newThreshold);
    setFilter(newFilter);
  };

  const handleRangeFilterChange = (min, max) => {
    // Reset single filter when using range filter
    setThreshold("");
    setFilter("all");
    // Set range filter values
    setMinThreshold(min);
    setMaxThreshold(max);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-secondary-900 to-primary-900">
      <Container maxWidth="lg" sx={{ pt: 12, pb: 4 }}>
        <Box sx={{ width: '100%' }}>
          <AppBar 
            position="static" 
            sx={{ 
              borderRadius: 1, 
              backgroundColor: 'rgba(30, 41, 59, 0.5)',
              backdropFilter: 'blur(8px)',
              border: '1px solid rgba(255, 255, 255, 0.1)'
            }}
          >
            <Tabs
              value={value}
              onChange={handleChange}
              indicatorColor="primary"
              textColor="primary"
              variant="fullWidth"
              sx={{
                '& .MuiTab-root': {
                  color: 'rgba(255, 255, 255, 0.7)',
                  '&.Mui-selected': {
                    color: 'white',
                  },
                },
                '& .MuiTabs-indicator': {
                  backgroundColor: 'primary.main',
                },
              }}
            >
              <Tab label="Attendance Analysis" />
              <Tab label="Marks Analysis" />
              <Tab label="Notify Students" />
            </Tabs>
          </AppBar>

          <TabPanel value={value} index={0}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3, 
                mt: 3, 
                backgroundColor: 'rgba(30, 41, 59, 0.5)',
                backdropFilter: 'blur(8px)',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}
            >
              <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={12}>
                  <Typography variant="h6" className="text-white mb-3">
                    Single Threshold Filter
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Attendance Threshold (%)"
                        type="number"
                        value={threshold}
                        onChange={(e) => handleSingleFilterChange(e.target.value, filter)}
                        inputProps={{ min: 0, max: 100 }}
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            color: 'white',
                            '& fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.23)',
                            },
                            '&:hover fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.5)',
                            },
                          },
                          '& .MuiInputLabel-root': {
                            color: 'rgba(255, 255, 255, 0.7)',
                          },
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>Filter</InputLabel>
                        <Select
                          value={filter}
                          label="Filter"
                          onChange={(e) => handleSingleFilterChange(threshold, e.target.value)}
                          sx={{
                            color: 'white',
                            '.MuiOutlinedInput-notchedOutline': {
                              borderColor: 'rgba(255, 255, 255, 0.23)',
                            },
                            '&:hover .MuiOutlinedInput-notchedOutline': {
                              borderColor: 'rgba(255, 255, 255, 0.5)',
                            },
                            '.MuiSvgIcon-root': {
                              color: 'white',
                            },
                          }}
                        >
                          <MenuItem value="all">All Students</MenuItem>
                          <MenuItem value="above">Above Threshold</MenuItem>
                          <MenuItem value="below">Below Threshold</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="h6" className="text-white mb-3">
                    Range Filter
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Min Threshold (%)"
                        type="number"
                        value={minThreshold}
                        onChange={(e) =>
                          handleRangeFilterChange(e.target.value, maxThreshold)
                        }
                        inputProps={{ min: 0, max: 100 }}
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            color: 'white',
                            '& fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.23)',
                            },
                            '&:hover fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.5)',
                            },
                          },
                          '& .MuiInputLabel-root': {
                            color: 'rgba(255, 255, 255, 0.7)',
                          },
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Max Threshold (%)"
                        type="number"
                        value={maxThreshold}
                        onChange={(e) =>
                          handleRangeFilterChange(minThreshold, e.target.value)
                        }
                        inputProps={{ min: 0, max: 100 }}
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            color: 'white',
                            '& fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.23)',
                            },
                            '&:hover fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.5)',
                            },
                          },
                          '& .MuiInputLabel-root': {
                            color: 'rgba(255, 255, 255, 0.7)',
                          },
                        }}
                      />
                    </Grid>
                  </Grid>
                </Grid>
              </Grid>

              <Typography variant="h5" className="text-white mb-4">
                {minThreshold && maxThreshold
                  ? `Students with Attendance Between ${minThreshold}% and ${maxThreshold}%`
                  : filter === "all"
                  ? "All Students Attendance"
                  : filter === "above"
                  ? `Students with Attendance Above ${threshold}%`
                  : `Students with Attendance Below ${threshold}%`}
              </Typography>

              {loading ? (
                <Box display="flex" justifyContent="center" p={3}>
                  <CircularProgress sx={{ color: 'white' }} />
                </Box>
              ) : attendanceData.length > 0 ? (
                <TableContainer>
                  <Table sx={{
                    '& .MuiTableCell-root': {
                      color: 'rgba(255, 255, 255, 0.8)',
                      borderColor: 'rgba(255, 255, 255, 0.1)',
                    },
                    '& .MuiTableCell-head': {
                      backgroundColor: 'rgba(30, 41, 59, 0.5)',
                      fontWeight: 'bold',
                    },
                    '& .MuiTableRow-root:nth-of-type(odd)': {
                      backgroundColor: 'rgba(30, 41, 59, 0.3)',
                    },
                    '& .MuiTableRow-root:nth-of-type(even)': {
                      backgroundColor: 'rgba(30, 41, 59, 0.5)',
                    },
                  }}>
                    <TableHead>
                      <TableRow>
                        <TableCell>Roll Number</TableCell>
                        <TableCell>Name</TableCell>
                        <TableCell align="right">Total Classes</TableCell>
                        <TableCell align="right">Attended Classes</TableCell>
                        <TableCell align="right">Attendance %</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {attendanceData.map((student) => (
                        <TableRow key={student.studentId}>
                          <TableCell>{student.rollNumber}</TableCell>
                          <TableCell>{student.name}</TableCell>
                          <TableCell align="right">
                            {student.totalSemesterClasses}
                          </TableCell>
                          <TableCell align="right">
                            {student.attendedClasses}
                          </TableCell>
                          <TableCell align="right">
                            {student.attendancePercentage.toFixed(2)}%
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Box display="flex" justifyContent="center" p={3}>
                  <Typography>No students found matching the criteria</Typography>
                </Box>
              )}
            </Paper>
          </TabPanel>

          <TabPanel value={value} index={1}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3, 
                mt: 3, 
                backgroundColor: 'rgba(30, 41, 59, 0.5)',
                backdropFilter: 'blur(8px)',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}
            >
              <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={3}>
                  <FormControl fullWidth>
                    <InputLabel sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>Subject</InputLabel>
                    <Select
                      value={subject}
                      label="Subject"
                      onChange={(e) => setSubject(e.target.value)}
                      sx={{
                        color: 'white',
                        '.MuiOutlinedInput-notchedOutline': {
                          borderColor: 'rgba(255, 255, 255, 0.23)',
                        },
                        '&:hover .MuiOutlinedInput-notchedOutline': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                        },
                        '.MuiSvgIcon-root': {
                          color: 'white',
                        },
                      }}
                    >
                      <MenuItem value="">All Subjects</MenuItem>
                      {subjectsList.map((subj) => (
                        <MenuItem key={subj.id} value={subj.name}>
                          {subj.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <TextField
                    fullWidth
                    label="Marks Threshold"
                    type="number"
                    value={marksThreshold}
                    onChange={(e) => setMarksThreshold(e.target.value)}
                    inputProps={{ min: 0, max: 100 }}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        color: 'white',
                        '& fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.23)',
                        },
                        '&:hover fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                        },
                      },
                      '& .MuiInputLabel-root': {
                        color: 'rgba(255, 255, 255, 0.7)',
                      },
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth>
                    <InputLabel sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>Filter</InputLabel>
                    <Select
                      value={marksFilter}
                      label="Filter"
                      onChange={(e) => setMarksFilter(e.target.value)}
                      sx={{
                        color: 'white',
                        '.MuiOutlinedInput-notchedOutline': {
                          borderColor: 'rgba(255, 255, 255, 0.23)',
                        },
                        '&:hover .MuiOutlinedInput-notchedOutline': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                        },
                        '.MuiSvgIcon-root': {
                          color: 'white',
                        },
                      }}
                    >
                      <MenuItem value="all">All Students</MenuItem>
                      <MenuItem value="above">Above Threshold</MenuItem>
                      <MenuItem value="below">Below Threshold</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <FormControl fullWidth>
                    <InputLabel sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>Exam Type</InputLabel>
                    <Select
                      value={examType}
                      label="Exam Type"
                      onChange={(e) => setExamType(e.target.value)}
                      sx={{
                        color: 'white',
                        '.MuiOutlinedInput-notchedOutline': {
                          borderColor: 'rgba(255, 255, 255, 0.23)',
                        },
                        '&:hover .MuiOutlinedInput-notchedOutline': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                        },
                        '.MuiSvgIcon-root': {
                          color: 'white',
                        },
                      }}
                    >
                      <MenuItem value="mid1">Mid Term 1</MenuItem>
                      <MenuItem value="mid2">Mid Term 2</MenuItem>
                      <MenuItem value="assignment1">Assignment 1</MenuItem>
                      <MenuItem value="assignment2">Assignment 2</MenuItem>
                      <MenuItem value="quiz">Quiz</MenuItem>
                      <MenuItem value="attendance">Attendance</MenuItem>
                      <MenuItem value="total">Total</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>

              <Typography variant="h5" className="text-white mb-4">
                {marksFilter === "all"
                  ? `${examType.toUpperCase()} Marks Analysis`
                  : marksFilter === "above"
                  ? `Students with ${examType.toUpperCase()} Marks Above ${marksThreshold}%`
                  : `Students with ${examType.toUpperCase()} Marks Below ${marksThreshold}%`}
              </Typography>

              {marksLoading ? (
                <Box display="flex" justifyContent="center" p={3}>
                  <CircularProgress sx={{ color: 'white' }} />
                </Box>
              ) : (
                <TableContainer>
                  <Table sx={{
                    '& .MuiTableCell-root': {
                      color: 'rgba(255, 255, 255, 0.8)',
                      borderColor: 'rgba(255, 255, 255, 0.1)',
                    },
                    '& .MuiTableCell-head': {
                      backgroundColor: 'rgba(30, 41, 59, 0.5)',
                      fontWeight: 'bold',
                    },
                    '& .MuiTableRow-root:nth-of-type(odd)': {
                      backgroundColor: 'rgba(30, 41, 59, 0.3)',
                    },
                    '& .MuiTableRow-root:nth-of-type(even)': {
                      backgroundColor: 'rgba(30, 41, 59, 0.5)',
                    },
                  }}>
                    <TableHead>
                      <TableRow>
                        <TableCell>Roll Number</TableCell>
                        <TableCell>Name</TableCell>
                        <TableCell align="right">Mid 1</TableCell>
                        <TableCell align="right">Mid 2</TableCell>
                        <TableCell align="right">Assignment 1</TableCell>
                        <TableCell align="right">Assignment 2</TableCell>
                        <TableCell align="right">Quiz</TableCell>
                        <TableCell align="right">Attendance</TableCell>
                        <TableCell align="right">Weighted Mid Marks</TableCell>
                        <TableCell align="right">Total Marks</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {marksData.map((student) => (
                        <TableRow key={student.studentId}>
                          <TableCell>{student.rollNumber}</TableCell>
                          <TableCell>{student.name}</TableCell>
                          <TableCell align="right">
                            {student.examMarks.mid1}
                          </TableCell>
                          <TableCell align="right">
                            {student.examMarks.mid2}
                          </TableCell>
                          <TableCell align="right">
                            {student.examMarks.assignment1}
                          </TableCell>
                          <TableCell align="right">
                            {student.examMarks.assignment2}
                          </TableCell>
                          <TableCell align="right">
                            {student.examMarks.quiz}
                          </TableCell>
                          <TableCell align="right">
                            {student.examMarks.attendance}
                          </TableCell>
                          <TableCell align="right">
                            {student.weightedMidMarks !== undefined
                              ? student.weightedMidMarks.toFixed(2)
                              : "N/A"}
                          </TableCell>
                          <TableCell align="right">
                            {student.totalMarks !== undefined
                              ? student.totalMarks.toFixed(2)
                              : "N/A"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Paper>
          </TabPanel>

          <TabPanel value={value} index={2}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3, 
                mt: 3, 
                backgroundColor: 'rgba(30, 41, 59, 0.5)',
                backdropFilter: 'blur(8px)',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}
            >
              <Typography variant="h5" className="text-white mb-4">
                Send Attendance Reports
              </Typography>

              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    multiline
                    rows={4}
                    label="Custom Message (Optional)"
                    value={customMessage}
                    onChange={(e) => setCustomMessage(e.target.value)}
                    variant="outlined"
                    placeholder="Enter a custom message to include in the email..."
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        color: 'white',
                        '& fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.23)',
                        },
                        '&:hover fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                        },
                      },
                      '& .MuiInputLabel-root': {
                        color: 'rgba(255, 255, 255, 0.7)',
                      },
                    }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="h6" className="text-white mb-4">
                    Quick Filters
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item>
                      <Button
                        variant="contained"
                        onClick={() => handleSendEmails()}
                      >
                        Send to All Students ({attendanceData.length})
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="contained"
                        color="warning"
                        onClick={() => handleSendEmails(75, "below")}
                      >
                        Send to Below 75% (
                        {
                          attendanceData.filter(
                            (s) => s.attendancePercentage < 75
                          ).length
                        }
                        )
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="contained"
                        color="success"
                        onClick={() => handleSendEmails(90, "above")}
                      >
                        Send to Above 90% (
                        {
                          attendanceData.filter(
                            (s) => s.attendancePercentage > 90
                          ).length
                        }
                        )
                      </Button>
                    </Grid>
                  </Grid>
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="h6" className="text-white mb-4">
                    Custom Range
                  </Typography>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} sm={4}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Min Percentage"
                        value={emailMinThreshold}
                        onChange={(e) => {
                          setEmailMinThreshold(e.target.value);
                          setEmailMaxThreshold("");
                        }}
                        inputProps={{ min: 0, max: 100 }}
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            color: 'white',
                            '& fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.23)',
                            },
                            '&:hover fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.5)',
                            },
                          },
                          '& .MuiInputLabel-root': {
                            color: 'rgba(255, 255, 255, 0.7)',
                          },
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={4}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Max Percentage"
                        value={emailMaxThreshold}
                        onChange={(e) => {
                          setEmailMaxThreshold(e.target.value);
                        }}
                        inputProps={{ min: 0, max: 100 }}
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            color: 'white',
                            '& fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.23)',
                            },
                            '&:hover fieldset': {
                              borderColor: 'rgba(255, 255, 255, 0.5)',
                            },
                          },
                          '& .MuiInputLabel-root': {
                            color: 'rgba(255, 255, 255, 0.7)',
                          },
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={4}>
                      <Button
                        variant="contained"
                        fullWidth
                        onClick={() =>
                          handleSendEmails(
                            undefined,
                            undefined,
                            Number(emailMinThreshold),
                            Number(emailMaxThreshold)
                          )
                        }
                        disabled={!emailMinThreshold || !emailMaxThreshold}
                        sx={{ height: "56px" }}
                      >
                        Send to Range{" "}
                        {emailMinThreshold && emailMaxThreshold
                          ? `(${
                              attendanceData.filter(
                                (s) =>
                                  s.attendancePercentage >=
                                    Number(emailMinThreshold) &&
                                  s.attendancePercentage <=
                                    Number(emailMaxThreshold)
                              ).length
                            } students)`
                          : ""}
                      </Button>
                    </Grid>
                  </Grid>
                </Grid>
              </Grid>

              <Divider sx={{ my: 4 }} />

              <Typography variant="h5" className="text-white mb-4">
                Send Marks Reports
              </Typography>

              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    multiline
                    rows={4}
                    label="Custom Message (Optional)"
                    value={customMessage}
                    onChange={(e) => setCustomMessage(e.target.value)}
                    variant="outlined"
                    placeholder="Enter a custom message to include in the email..."
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        color: 'white',
                        '& fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.23)',
                        },
                        '&:hover fieldset': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                        },
                      },
                      '& .MuiInputLabel-root': {
                        color: 'rgba(255, 255, 255, 0.7)',
                      },
                    }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="h6" className="text-white mb-4">
                    Send Options
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item>
                      <Button
                        variant="contained"
                        onClick={() => handleSendMarksEmails()}
                      >
                        Send to All Students
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="contained"
                        color="warning"
                        onClick={() =>
                          handleSendMarksEmails(20, "below", "total")
                        }
                      >
                        Send to Below 20 Total
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="contained"
                        color="error"
                        onClick={() => handleSendMarksEmails(13, "below", "mid1")}
                      >
                        Send to Below 13 in Mid 1 for 20
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="contained"
                        color="success"
                        onClick={() => handleSendMarksEmails(15, "above", "mid2")}
                      >
                        Send to Above 15 in Mid 2 for 20
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="contained"
                        color="error"
                        onClick={() =>
                          handleSendMarksEmails(0, "below", "assignment1")
                        }
                      >
                        Send to Zero Marks Students (Assignment 1)
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="contained"
                        color="error"
                        onClick={() =>
                          handleSendMarksEmails(0, "below", "assignment2")
                        }
                      >
                        Send to Zero Marks Students (Assignment 2)
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="contained"
                        color="error"
                        onClick={() => handleSendMarksEmails(0, "below", "quiz")}
                      >
                        Send to Zero Marks Students (Quiz)
                      </Button>
                    </Grid>
                  </Grid>
                </Grid>
              </Grid>
            </Paper>
          </TabPanel>
        </Box>

        <Snackbar 
          open={openSnackbar} 
          autoHideDuration={6000} 
          onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert 
            onClose={handleCloseSnackbar} 
            severity={emailStatus?.severity || 'info'} 
            sx={{ 
              backgroundColor: 'rgba(30, 41, 59, 0.9)',
              color: 'white',
              '.MuiAlert-icon': {
                color: 'white',
              },
            }}
          >
            {emailStatus?.message}
          </Alert>
        </Snackbar>
      </Container>
    </div>
  );
}

export default Proctordashboard;
