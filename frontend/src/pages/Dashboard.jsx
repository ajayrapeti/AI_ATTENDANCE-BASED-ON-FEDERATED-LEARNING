import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import Calendar from '../components/calender/Calender';
import dummyData from '../components/dummydata/dummydata';
import axios from 'axios';
import { useSession } from '../contexts/SessionContext';
import { Container, Box, AppBar, Tabs, Tab, Paper, Grid, Typography, Button, CircularProgress, Alert, Snackbar, IconButton } from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import DeleteIcon from '@mui/icons-material/Delete';

const calculateMidAvg = (mid1, mid2) => {
  const [highest, lowest] = [mid1, mid2].sort((a, b) => b - a);
  return Math.ceil((highest * 2/3) + (lowest * 1/3));
};

const calculateTotal = (midAvg, assignAvg = 0, quiz = 0, attendance = 0) => {
  return Math.ceil(parseFloat(midAvg) + parseFloat(assignAvg) + quiz + attendance);
};

// TabPanel component
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
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

const Dashboard = () => {
  const { session } = useSession();
  const [studentData, setStudentData] = useState(null);
  const [value, setValue] = useState(0);
  const [images, setImages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });
  const [isCameraActive, setIsCameraActive] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [captureCount, setCaptureCount] = useState(0);
  const [captureProgress, setCaptureProgress] = useState(0);
  const [autoCaptureActive, setAutoCaptureActive] = useState(false);
  const captureIntervalRef = useRef(null);
  const [uploadMode, setUploadMode] = useState('camera'); // 'camera' or 'folder'
  const [isUploadingFolder, setIsUploadingFolder] = useState(false);

  useEffect(() => {
    const fetchStudentData = async () => {
      try {
        const response = await axios.get('http://localhost:3000/auth/profile/details', {
          headers: {
            Authorization: `Bearer ${session.token}`
          }
        });
        setStudentData(response.data.data);
      } catch (error) {
        console.error('Failed to fetch student data:', error);
      }
    };
    console.log(session);

    if (session?.token) {
      fetchStudentData();
    }
  }, [session]);

  const formatAttendanceForCalendar = (attendance) => {
    const formattedAttendance = {};
    
    attendance?.forEach(entry => {
      if (!formattedAttendance[entry.date]) {
        formattedAttendance[entry.date] = [];
      }
      formattedAttendance[entry.date].push({
        period: entry.period,
        present: entry.present,
        subject: entry.subject.name
      });
    });

    return formattedAttendance;
  };

  const formatEventsForCalendar = (events) => {
    const holidays = [];
    const exams = [];
    const activities = [];

    events?.forEach(event => {
      switch(event.type) {
        case 'holiday':
          holidays.push({ date: event.date, title: event.title });
          break;
        case 'exam':
          exams.push({ date: event.date, title: event.title });
          break;
        case 'activity':
          activities.push({ date: event.date, title: event.title });
          break;
        default:
          break;
      }
    });

    return {
      holidays,
      weeklyHolidays: ['Sunday'],
      exams,
      activities
    };
  };

  const handleChange = (event, newValue) => {
    setValue(newValue);
    if (newValue !== 3 && streamRef.current) {
      stopCamera();
    }
  };

  const startCamera = async () => {
    try {
      if (!videoRef.current) {
        console.error("Video ref is null - waiting for component to mount");
        setNotification({
          open: true,
          message: 'Camera element not ready yet. Please try again.',
          severity: 'warning'
        });
        return;
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play().catch(e => console.error("Error playing video:", e));
        setIsCameraActive(true);
        console.log("Camera started, dimensions:", videoRef.current.videoWidth, videoRef.current.videoHeight);
      };
      streamRef.current = stream;
    } catch (error) {
      console.error("Error accessing camera:", error);
      setNotification({
        open: true,
        message: 'Error accessing camera: ' + error.message,
        severity: 'error'
      });
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsCameraActive(false);
    }
  };

  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) {
      console.error("Video or canvas ref is null");
      setNotification({
        open: true,
        message: 'Camera not ready yet',
        severity: 'error'
      });
      return;
    }
    
    setIsCapturing(true);
    
    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      console.log("Video dimensions:", video.videoWidth, video.videoHeight);
      
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      const base64Image = canvas.toDataURL('image/jpeg');
      console.log("Image captured, length:", base64Image.length);
      
      const timestamp = new Date().toISOString();
      setImages(prevImages => [
        ...prevImages, 
        { 
          name: `capture_${timestamp}.jpg`,
          base64: base64Image,
          timestamp
        }
      ]);
      
      setCaptureCount(prev => prev + 1);
      
      setNotification({
        open: true,
        message: `Image ${captureCount + 1} captured successfully`,
        severity: 'success'
      });
    } catch (error) {
      console.error("Error capturing image:", error);
      setNotification({
        open: true,
        message: 'Error capturing image: ' + error.message,
        severity: 'error'
      });
    } finally {
      setTimeout(() => {
        setIsCapturing(false);
      }, 500);
    }
  };

  const startAutoCapture = () => {
    if (autoCaptureActive) {
      stopAutoCapture();
      return;
    }
    
    setAutoCaptureActive(true);
    setCaptureProgress(0);
    
    captureIntervalRef.current = setInterval(() => {
      captureImage();
      setCaptureProgress(prev => {
        const newProgress = prev + 1;
        if (newProgress >= 100) {
          stopAutoCapture();
        }
        return newProgress;
      });
    }, 2000);
  };
  
  const stopAutoCapture = () => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    setAutoCaptureActive(false);
    setCaptureProgress(0);
  };

  const deleteImage = (index) => {
    setImages(prevImages => prevImages.filter((_, i) => i !== index));
  };

  const generateBase64AndSend = async () => {
    if (images.length === 0) {
      setNotification({
        open: true,
        message: 'No images to process',
        severity: 'warning'
      });
      return;
    }

    setIsProcessing(true);

    try {
      // Get student roll number from studentData
      const studentRollNo = studentData?.student?.rollNo;
      if (!studentRollNo) {
        throw new Error('Student Roll Number not found');
      }

      // Prepare data in the format expected by the API
      const requestData = {
        student_folder: studentRollNo,
        student_id: studentRollNo,
        images: images.map((img, index) => ({
          filename: `${studentRollNo}_${index + 1}.jpg`,
          base64_string: img.base64.split(',')[1]
        }))
      };

      // Send to API
      const response = await axios.post(
        'http://localhost:8000/api/aggregation/process_base64_images/',
        requestData,
        {
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${session.token}`
          }
        }
      );

      if (response.data.code === 200) {
        setNotification({
          open: true,
          message: `Embeddings generated successfully for student ${studentRollNo}`,
          severity: 'success'
        });

        // Optional: Clear images after successful processing
        if (window.confirm('Images processed successfully. Clear captured images?')) {
          setImages([]);
        }
      } else {
        throw new Error(response.data.msg || 'Failed to process images');
      }

    } catch (error) {
      console.error('Error processing images:', error);
      setNotification({
        open: true,
        message: error.response?.data?.msg || error.message || 'Error processing images',
        severity: 'error'
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
    };
  }, []);

  // Handle folder upload
  const handleFolderUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIsUploadingFolder(true);
    setImages([]); // Clear existing images

    try {
      const newImages = [];
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.type.startsWith('image/')) {
          const reader = new FileReader();
          const promise = new Promise((resolve) => {
            reader.onload = (e) => {
              const base64 = e.target.result;
              newImages.push({
                name: file.name,
                base64: base64,
                timestamp: new Date().toISOString()
              });
              resolve();
            };
          });
          reader.readAsDataURL(file);
          await promise;
        }
      }

      setImages(newImages);
      setNotification({
        open: true,
        message: `Successfully uploaded ${newImages.length} images`,
        severity: 'success'
      });
    } catch (error) {
      console.error('Error uploading folder:', error);
      setNotification({
        open: true,
        message: 'Error uploading folder',
        severity: 'error'
      });
    } finally {
      setIsUploadingFolder(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-secondary-900 to-primary-900">
      <Container 
        maxWidth={false}
        sx={{ 
          pt: 12, 
          pb: 4,
          px: { xs: 2, sm: 3, md: 4 },
          maxWidth: '1300px !important',
          mx: 'auto',
        }}
      >
        {/* Student Info Card - Updated styling */}
        <Paper
          elevation={3}
          sx={{
            p: 2.5,
            mb: 3,
            backgroundColor: 'rgba(30, 41, 59, 0.7)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            borderRadius: '12px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.2)',
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Grid 
              container 
              spacing={2}
              alignItems="center"
            >
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" className="text-gray-400 text-sm">
                  Student Name
                </Typography>
                <Typography variant="h5" className="text-white font-semibold">
                  {studentData?.student?.name || 'Loading...'}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="subtitle2" className="text-gray-400 text-sm">
                  Roll Number
                </Typography>
                <Typography variant="body1" className="text-white font-semibold">
                  {studentData?.student?.rollNo || 'Loading...'}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="subtitle2" className="text-gray-400 text-sm">
                  Class
                </Typography>
                <Typography variant="body1" className="text-white font-semibold">
                  {studentData?.student?.class || 'Loading...'}
                </Typography>
              </Grid>
            </Grid>
          </motion.div>
        </Paper>

        {/* Existing Tabs and Content */}
        <Box sx={{ width: '100%' }}>
          <AppBar 
            position="static" 
            sx={{ 
              borderRadius: 1, 
              backgroundColor: 'rgba(30, 41, 59, 0.5)',
              backdropFilter: 'blur(8px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              width: '100%',
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
              <Tab label="Attendance Calendar" />
              <Tab label="Academic Performance" />
              <Tab label="Teacher Remarks" />
              <Tab label="Images" />
            </Tabs>
          </AppBar>

          {/* Calendar Tab */}
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
              <Calendar 
                attendance={formatAttendanceForCalendar(studentData?.attendance)}
                events={formatEventsForCalendar(studentData?.events)}
                studentId={studentData?.student?.rollNo}
              />
            </Paper>
          </TabPanel>

          {/* Academic Performance Tab */}
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
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <h2 className="text-2xl font-bold text-primary-100 mb-4">Academic Performance</h2>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-secondary-700">
                    <thead className="bg-secondary-800/50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Subject</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Mid 1 (20)</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Mid 2 (20)</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Assignment 1 (10)</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Assignment 2 (10)</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Quiz (5)</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Attendance (5)</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Total</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-secondary-700">
                      {studentData?.marks?.map((subject, index) => {
                        const mid1 = subject.marks.mid1 ? parseFloat(subject.marks.mid1) : 0;
                        const mid2 = subject.marks.mid2 ? parseFloat(subject.marks.mid2) : 0;
                        const assignment1 = subject.marks.assignment1 ? parseFloat(subject.marks.assignment1) : 0;
                        const assignment2 = subject.marks.assignment2 ? parseFloat(subject.marks.assignment2) : 0;
                        const quiz = subject.marks.quiz ? parseFloat(subject.marks.quiz) : 0;
                        const attendance = subject.marks.attendance ? parseFloat(subject.marks.attendance) : 0;

                        const midAvg = (mid1 || mid2) ? 
                          (mid1 && mid2 ? calculateMidAvg(mid1, mid2) : (mid1 || mid2)) : 
                          0;

                        const calculatedTotal = calculateTotal(midAvg, assignment1, assignment2, quiz, attendance);

                        return (
                          <tr key={subject.subjectId} className={index % 2 === 0 ? 'bg-secondary-800/50' : 'bg-secondary-700/50'}>
                            <td className="px-4 py-4 whitespace-nowrap text-sm font-medium text-gray-500">
                              {subject.subjectName}
                            </td>
                            <td className="px-4 py-4 text-center text-sm text-gray-500">
                              {subject.marks.mid1 || '-'}
                            </td>
                            <td className="px-4 py-4 text-center text-sm text-gray-500">
                              {subject.marks.mid2 || '-'}
                            </td>
                            <td className="px-4 py-4 text-center text-sm text-gray-500">
                              {subject.marks.assignment1 || '-'}
                            </td>
                            <td className="px-4 py-4 text-center text-sm text-gray-500">
                              {subject.marks.assignment2 || '-'}
                            </td>
                            <td className="px-4 py-4 text-center text-sm text-gray-500">
                              {quiz || '-'}
                            </td>
                            <td className="px-4 py-4 text-center text-sm text-gray-500">
                              {attendance || '-'}
                            </td>
                            <td className="px-4 py-4 text-center text-sm font-medium text-green-600 font-bold">
                              {calculatedTotal}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            </Paper>
          </TabPanel>

          {/* Teacher Remarks Tab */}
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
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <h2 className="text-2xl font-bold text-primary-100 mb-4">Teacher Remarks</h2>
                <div className="space-y-4">
                  {dummyData.remarks.map((remark, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.2 }}
                      className="border-l-4 border-primary-500 pl-4 py-2"
                    >
                      <p className="text-secondary-300">{remark.comment}</p>
                      <div className="mt-2 flex justify-between text-sm text-secondary-400">
                        <span>{remark.teacher}</span>
                        <span>{remark.date}</span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            </Paper>
          </TabPanel>

          {/* Images Tab */}
          <TabPanel value={value} index={3}>
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
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <h2 className="text-2xl font-bold text-primary-100 mb-4">Image Capture</h2>

                {/* Mode Selection */}
                <div className="flex gap-4 mb-6">
                  {/* <Button
                    variant={uploadMode === 'camera' ? 'contained' : 'outlined'}
                    onClick={() => setUploadMode('camera')}
                    startIcon={<CameraAltIcon />}
                  >
                    Use Camera
                  </Button> */}
                  {/* <Button
                    variant={uploadMode === 'folder' ? 'contained' : 'outlined'}
                    component="label"
                    startIcon={<FolderIcon />}
                  >
                    Upload Folder
                    <input
                      type="file"
                      hidden
                      multiple
                      webkitdirectory="true"
                      onChange={handleFolderUpload}
                    />
                  </Button> */}
                </div>

                {/* Camera Mode */}
                {uploadMode === 'camera' && (
                  <>
                    <div className="flex flex-wrap gap-4">
                      <Button
                        variant="contained"
                        startIcon={<CameraAltIcon />}
                        onClick={isCameraActive ? stopCamera : startCamera}
                        sx={{ 
                          bgcolor: isCameraActive ? 'error.main' : 'primary.main',
                          '&:hover': { bgcolor: isCameraActive ? 'error.dark' : 'primary.dark' }
                        }}
                      >
                        {isCameraActive ? 'Stop Camera' : 'Start Camera'}
                      </Button>
                      
                      {isCameraActive && (
                        <>
                          <Button
                            variant="contained"
                            onClick={captureImage}
                            disabled={isCapturing || autoCaptureActive}
                            sx={{ 
                              bgcolor: 'secondary.main',
                              '&:hover': { bgcolor: 'secondary.dark' }
                            }}
                          >
                            {isCapturing ? 'Capturing...' : 'Take Photo'}
                          </Button>
                          
                          <Button
                            variant="contained"
                            onClick={startAutoCapture}
                            disabled={isCapturing}
                            color={autoCaptureActive ? "error" : "info"}
                            sx={{ 
                              position: 'relative',
                              overflow: 'hidden'
                            }}
                          >
                            {autoCaptureActive ? 'Stop Auto Capture' : 'Start Auto Capture'}
                            {autoCaptureActive && (
                              <Box 
                                sx={{
                                  position: 'absolute',
                                  bottom: 0,
                                  left: 0,
                                  height: '4px',
                                  bgcolor: 'rgba(255,255,255,0.5)',
                                  width: `${captureProgress}%`,
                                  transition: 'width 0.5s'
                                }}
                              />
                            )}
                          </Button>
                        </>
                      )}
                    </div>
                  </>
                )}

                {/* Folder Upload Mode */}
                {uploadMode === 'folder' && (
                  <div className="space-y-4">
                    {isUploadingFolder ? (
                      <div className="flex items-center gap-2">
                        <CircularProgress size={20} />
                        <span>Uploading images...</span>
                      </div>
                    ) : (
                      <Typography variant="body1">
                        {images.length > 0 
                          ? `Uploaded ${images.length} images`
                          : 'No images uploaded yet'}
                      </Typography>
                    )}
                  </div>
                )}

                {isCameraActive && (
                  <div className="flex items-center gap-4">
                    <div className={`h-3 w-3 rounded-full ${isCapturing ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></div>
                    <span className="text-sm text-gray-300">
                      {isCapturing ? 'Capturing...' : 'Camera ready'}
                      {autoCaptureActive && ` (Auto-capturing: ${captureProgress}/100)`}
                    </span>
                  </div>
                )}
                
                {isCameraActive && (
                  <div className="mt-4 relative">
                    <div className={`bg-black rounded-lg overflow-hidden aspect-video max-w-2xl mx-auto border-2 ${isCapturing ? 'border-red-500 animate-pulse' : 'border-transparent'}`}>
                      {streamRef.current && (
                        <video 
                          autoPlay 
                          playsInline 
                          muted
                          className="w-full h-full object-contain"
                          ref={(el) => {
                            if (el && streamRef.current) {
                              el.srcObject = streamRef.current;
                            }
                          }}
                        />
                      )}
                      {isCapturing && (
                        <div className="absolute inset-0 bg-red-500 bg-opacity-20 flex items-center justify-center">
                          <div className="text-white text-xl font-bold">Capturing...</div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                {isProcessing && (
                  <div className="flex justify-center my-4">
                    <CircularProgress color="primary" />
                  </div>
                )}
                
                {images.length > 0 && (
                  <div className="mt-6">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className="text-xl font-semibold text-primary-100">
                        Captured Images ({images.length})
                      </h3>
                      <div className="flex gap-2">
                        <Button 
                          variant="contained"
                          color="info"
                          startIcon={<SaveIcon />}
                          onClick={generateBase64AndSend}
                          disabled={isProcessing || images.length === 0 || autoCaptureActive}
                          sx={{ 
                            bgcolor: 'info.main',
                            '&:hover': { bgcolor: 'info.dark' }
                          }}
                        >
                          {isProcessing ? 'Processing...' : 'Process Images'}
                        </Button>
                        
                        <Button 
                          variant="outlined" 
                          color="error" 
                          size="small"
                          onClick={() => setImages([])}
                          disabled={autoCaptureActive || isProcessing}
                        >
                          Clear All
                        </Button>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                      {images.map((img, index) => (
                        <div 
                          key={index} 
                          className="relative bg-secondary-800 rounded-lg overflow-hidden group"
                        >
                          <img 
                            src={img.base64} 
                            alt={`Captured ${index}`}
                            className="w-full h-32 object-cover"
                          />
                          <div className="p-2 text-xs text-gray-400 truncate flex justify-between items-center">
                            <span>Image {index + 1}</span>
                            <IconButton 
                              size="small" 
                              color="error"
                              onClick={() => deleteImage(index)}
                              disabled={autoCaptureActive || isProcessing}
                              className="opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            </Paper>
          </TabPanel>
        </Box>
      </Container>

      {/* Notification Snackbar */}
      <Snackbar 
        open={notification.open} 
        autoHideDuration={6000} 
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>

      {/* Always render the video and canvas elements, but keep them hidden when not in use */}
      <div style={{ display: 'none' }}>
        <video ref={videoRef} autoPlay playsInline muted />
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
};

export default Dashboard;