from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from core.models import Student, GlobalModel, TestImage
from django.conf import settings
import numpy as np
import os
from datetime import datetime
import cv2
from insightface.app import FaceAnalysis
import logging
import base64
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlobalModelTester:
    def __init__(self, model_dir="data/global_model", test_dir="data/test-images"):
        """Initialize with model and test directories"""
        self.model_dir = model_dir
        self.test_dir = test_dir
        
        # Initialize face processor
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_latest_model(self):
        """Load the latest global model"""
        try:
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("global_model_")]
            if not model_files:
                self.logger.error("No global model files found!")
                return None
            
            # Get most recent file
            latest_model = max(model_files)
            model_path = os.path.join(self.model_dir, latest_model)
            
            # Load model data
            model_data = np.load(model_path, allow_pickle=True)
            return model_data['student_embeddings'].item()
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None

    def process_test_image(self, image_path):
        """Process a single test image"""
        try:
            # Read and process image
            img = cv2.imread(image_path)
            if img is None:
                return None, None, "Failed to read image"
            
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get faces
            faces = self.app.get(img)
            if not faces:
                return None, None, "No faces detected"
            
            # Process each face
            processed_faces = []
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                processed_faces.append({
                    'bbox': bbox,
                    'embedding': embedding,
                    'score': face.det_score
                })
            
            return processed_faces, img, None
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return None, None, str(e)

    def compare_embeddings(self, test_embedding, student_embeddings):
        """Compare embeddings and return matches"""
        matches = {}
        try:
            # Normalize test embedding
            test_embedding = test_embedding / np.linalg.norm(test_embedding)
            
            # Compare with each student
            for student_id, student_embedding in student_embeddings.items():
                # Normalize student embedding
                student_embedding = student_embedding / np.linalg.norm(student_embedding)
                
                # Calculate similarity
                similarity = np.dot(test_embedding, student_embedding)
                matches[student_id] = similarity
                
                # Log high matches
                if similarity > 0.5:
                    self.logger.info(f"Face matched with Student {student_id} (similarity: {similarity:.2f})")
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Error comparing embeddings: {str(e)}")
            return {}

    def draw_face_boxes(self, image, faces, match_results):
        """Draw boxes and labels on faces"""
        print("DEBUG: Inside GlobalModelTester.draw_face_boxes")
        try:
            img = image.copy()
            print(f"DEBUG: Processing {len(faces)} faces with {len(match_results)} match results")
            
            for face, matches in zip(faces, match_results):
                x1, y1, x2, y2 = face['bbox']
                print(f"DEBUG: Drawing box at coordinates: ({x1}, {y1}, {x2}, {y2})")
                
                # Find best match
                if matches:
                    best_match = max(matches.items(), key=lambda x: x[1])
                    student_id, similarity = best_match
                    print(f"DEBUG: Best match - Student ID: {student_id}, Similarity: {similarity}")
                    
                    # Extract last 3 digits
                    last_three_digits = student_id[-3:]
                    print(f"DEBUG: Displaying last three digits: {last_three_digits}")
                    
                    if similarity > 0.5:
                        # Green box for match
                        color = (0, 255, 0)
                        # Draw large roll number (last 3 digits)
                        font_scale = 2.0  # Much larger font size
                        font_thickness = 4  # Thicker font
                        
                        # Calculate text size for centering
                        (text_width, text_height), _ = cv2.getTextSize(
                            last_three_digits, 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            font_thickness
                        )
                        
                        # Position the text above the box
                        text_x = x1 + (x2 - x1 - text_width) // 2  # Center horizontally
                        text_y = y1 - 20  # Position above the box
                        
                        # Draw box and large roll number
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            img, 
                            last_three_digits,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            color,
                            font_thickness
                        )
                        
                        # Draw smaller similarity score
                        similarity_text = f"({similarity:.2f})"
                        cv2.putText(
                            img,
                            similarity_text,
                            (text_x, text_y + text_height + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            1
                        )
                    else:
                        # Red box for low similarity
                        color = (0, 0, 255)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            img,
                            "Low match",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )
                else:
                    # Red box for no matches
                    color = (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img,
                        "No match",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
            
            return img
            
        except Exception as e:
            print(f"DEBUG: Error in GlobalModelTester.draw_face_boxes: {str(e)}")
            logger.error(f"Error drawing result: {str(e)}")
            return image

class AggregationViewSet(viewsets.ViewSet):
    parser_classes = (JSONParser, MultiPartParser, FormParser)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_dir = os.path.join(settings.MEDIA_ROOT, 'global_model')
        test_dir = os.path.join(settings.MEDIA_ROOT, 'test_images')
        self.tester = GlobalModelTester(model_dir=model_dir, test_dir=test_dir)
        self.logger = logging.getLogger(__name__)
        
        # Initialize face processor
        try:
            self.face_app = FaceAnalysis(
                name="buffalo_l",
                root="model",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=(320, 320))
            self.logger.info("Face processor initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing face processor: {str(e)}")
            raise

    @action(detail=False, methods=['POST'])
    def test_image(self, request):
        """Test an image against the global model"""
        try:
            print("DEBUG: Starting test_image endpoint")
            # Get image from request
            image_file = request.FILES.get('image')
            if not image_file:
                return Response({
                    'code': 400,
                    'msg': 'No image provided',
                    'data': None
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Save test image
            test_image = TestImage.objects.create(image=image_file)
            image_path = test_image.image.path
            print(f"DEBUG: Image saved to {image_path}")
            
            # Process image
            faces, img, error = self.tester.process_test_image(image_path)
            print(f"DEBUG: Processed image, found {len(faces) if faces else 0} faces")
            if error:
                return Response({
                    'code': 400,
                    'msg': error,
                    'data': None
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Load model
            student_embeddings = self.tester.load_latest_model()
            print(f"DEBUG: Loaded model with {len(student_embeddings) if student_embeddings else 0} embeddings")
            
            # Compare each face
            match_results = []
            for face in faces:
                matches = self.tester.compare_embeddings(face['embedding'], student_embeddings)
                match_results.append(matches)
                print(f"DEBUG: Found {len(matches)} matches for face")
            
            # Draw results
            print("DEBUG: About to draw face boxes")
            result_img = self.tester.draw_face_boxes(img, faces, match_results)
            print("DEBUG: Face boxes drawn")
            
            # Save result image
            result_filename = f"result_{os.path.basename(image_path)}"
            result_path = os.path.join(settings.MEDIA_ROOT, 'test_results', result_filename)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"DEBUG: Result image saved to {result_path}")
            
            # Prepare clean response
            clean_results = []
            seen_students = set()  # Track unique students
            total_students = len([k for k in student_embeddings.keys() if not k.endswith('_embedding')])  # Count unique students
            
            for face, matches in zip(faces, match_results):
                if matches:
                    # Sort matches by similarity
                    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
                    # Get matches above threshold and remove duplicates
                    for student_id, similarity in sorted_matches:
                        # Skip if already seen or if it's a duplicate (_embedding suffix)
                        if student_id in seen_students or student_id.endswith('_embedding'):
                            continue
                        
                        if similarity > 0.5:  # Threshold check
                            clean_results.append({
                                'student_id': student_id,
                                'similarity': float(similarity)
                            })
                            seen_students.add(student_id)
            
            # Sort final results by similarity
            clean_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Update test image record
            test_image.result = clean_results
            test_image.result_image = f'test_results/{result_filename}'
            test_image.processed = True
            test_image.save()
            
            return Response({
                'code': 200,
                'msg': 'Image processed successfully',
                'data': {
                    'total_students_in_database': total_students,
                    'total_matches': len(clean_results),
                    'matches': clean_results,
                    'result_image_url': f'/media/test_results/{result_filename}'
                }
            })
            
        except Exception as e:
            logger.error(f"Error in test_image: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return Response({
                'code': 500,
                'msg': str(e),
                'data': None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def load_latest_model(self):
        """Load the latest global model with proper error handling"""
        try:
            # Get latest model from database
            latest_model = GlobalModel.objects.latest('created_at')
            logger.info(f"Found model in database: {latest_model.version}")
            
            # Try different possible paths
            possible_paths = [
                os.path.join(settings.MEDIA_ROOT, latest_model.model_file.name),  # Full path
                os.path.join(settings.MEDIA_ROOT, 'global_model', latest_model.model_file.name),  # In global_model dir
                os.path.join(settings.MEDIA_ROOT, 'global_model', os.path.basename(latest_model.model_file.name))  # Just filename
            ]
            
            # Try each path
            model_path = None
            for path in possible_paths:
                logger.info(f"Checking path: {path}")
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Found model file at: {path}")
                    break
            
            if not model_path:
                logger.error("Model file not found in any location")
                return {}
            
            # Load the model
            model_data = np.load(model_path, allow_pickle=True)
            embeddings = model_data['student_embeddings'].item()
            
            # Convert all embeddings to float32
            for student_id in embeddings:
                embeddings[student_id] = np.array(embeddings[student_id]).astype(np.float32)
            
            logger.info(f"Successfully loaded model with {len(embeddings)} students")
            return embeddings
            
        except GlobalModel.DoesNotExist:
            logger.error("No model found in database")
            return {}
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def get_face_embedding(self, image_path):
        """Get face embedding using exact implementation from test_global_model.py"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None, None, "Could not load image"
            
            # Convert BGR to RGB (important!)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get faces using buffalo_l model
            faces = self.tester.app.get(img)
            if not faces:
                return None, None, "No faces detected in test image"
            
            # Get largest face as in test_global_model.py
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Get face info for drawing
            x1, y1, x2, y2 = map(int, face.bbox[:4])
            faces_info = [{
                'bbox': [x1, y1, x2, y2],
                'score': face.det_score
            }]
            
            # Get embedding and normalize exactly as in test_global_model.py
            embedding = face.embedding.astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Log embedding stats
            logger.info(f"Face detection score: {face.det_score}")
            logger.info(f"Embedding norm: {np.linalg.norm(embedding)}")
            
            return embedding, faces_info, None
            
        except Exception as e:
            logger.error(f"Error processing test image: {str(e)}")
            return None, None, str(e)

    def compare_embeddings(self, test_embedding, student_embeddings):
        """Compare embeddings using exact implementation from test_global_model.py"""
        results = []
        try:
            # Remove duplicate student IDs (remove ones with '_embedding' suffix)
            clean_embeddings = {
                student_id: emb for student_id, emb in student_embeddings.items() 
                if not student_id.endswith('_embedding')
            }
            
            for student_id, student_embedding in clean_embeddings.items():
                # Normalize student embedding
                student_embedding = student_embedding.astype(np.float32)
                student_embedding = student_embedding / np.linalg.norm(student_embedding)
                
                # Calculate cosine similarity exactly as in test_global_model.py
                similarity = float(np.dot(test_embedding, student_embedding))
                
                results.append({
                    'student_id': student_id,
                    'similarity': similarity
                })
                
                # Log matches as in test_global_model.py
                if similarity > 0.5:
                    logger.info(f"Face matched with Student {student_id} (similarity: {similarity:.2f})")
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            return []

    def aggregate_embeddings(self, new_embeddings):
        """Aggregate embeddings using server implementation"""
        try:
            # Load existing embeddings
            existing_embeddings = self.load_latest_model()
            
            # Convert and normalize all embeddings
            all_embeddings = {}
            
            # Process existing embeddings
            for student_id, embedding in existing_embeddings.items():
                try:
                    embedding = np.array(embedding, dtype=np.float32)
                    normalized_embedding = embedding / np.linalg.norm(embedding)
                    all_embeddings[student_id] = normalized_embedding
                except Exception as e:
                    logger.error(f"Error processing existing embedding for {student_id}: {str(e)}")
                    continue
            
            # Process new embeddings
            for student_id, embedding in new_embeddings.items():
                try:
                    embedding = np.array(embedding, dtype=np.float32)
                    normalized_embedding = embedding / np.linalg.norm(embedding)
                    all_embeddings[student_id] = normalized_embedding
                except Exception as e:
                    logger.error(f"Error processing new embedding for {student_id}: {str(e)}")
                    continue
            
            # Save new model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"global_model_{timestamp}.npz"
            model_path = os.path.join(self.model_dir, model_filename)
            
            # Save with all normalized embeddings
            np.savez(
                model_path,
                student_embeddings=all_embeddings,
                num_students=len(all_embeddings),
                timestamp=timestamp
            )
            
            # Log aggregation results
            logger.info(f"Aggregated {len(all_embeddings)} total embeddings")
            logger.info(f"Existing: {len(existing_embeddings)}, New: {len(new_embeddings)}")
            
            # Create database entry
            relative_path = os.path.join('global_model', model_filename)
            global_model = GlobalModel.objects.create(
                version=timestamp,
                model_file=relative_path,
                num_students=len(all_embeddings)
            )
            
            return global_model, None, len(existing_embeddings)
            
        except Exception as e:
            logger.error(f"Error in aggregate_embeddings: {str(e)}")
            return None, str(e), 0

    @action(detail=False, methods=['POST'])
    def receive_embeddings(self, request):
        """Receive embeddings using server implementation"""
        try:
            files = request.FILES.getlist('embedding_files')
            if not files:
                return Response({
                    'code': 400,
                    'msg': 'No embedding files provided',
                    'data': None
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Process embeddings
            new_embeddings = {}
            for file in files:
                try:
                    # Get student ID from filename (remove .npy extension)
                    student_id = os.path.splitext(file.name)[0]
                    
                    # Save file
                    file_path = os.path.join(settings.MEDIA_ROOT, 'embeddings', file.name)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'wb+') as destination:
                        for chunk in file.chunks():
                            destination.write(chunk)
                    
                    # Load and normalize embedding
                    embedding = np.load(file_path)
                    normalized_embedding = embedding / np.linalg.norm(embedding)
                    new_embeddings[student_id] = normalized_embedding
                    
                    # Update student record
                    Student.objects.update_or_create(
                        student_id=student_id,
                        defaults={'embedding_file': f'embeddings/{file.name}'}
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing file {file.name}: {str(e)}")
                    return Response({
                        'code': 400,
                        'msg': f'Error processing file {file.name}',
                        'data': {'error': str(e)}
                    }, status=status.HTTP_400_BAD_REQUEST)
            
            # Aggregate embeddings
            global_model, error, existing_count = self.aggregate_embeddings(new_embeddings)
            
            if error:
                return Response({
                    'code': 500,
                    'msg': 'Error during aggregation',
                    'data': {'error': error}
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            return Response({
                'code': 200,
                'msg': 'Embeddings received and model updated successfully',
                'data': {
                    'model_id': global_model.id,
                    'total_students': global_model.num_students,
                    'existing_students': existing_count,
                    'new_students': len(new_embeddings),
                    'processed_files': [f.name for f in files]
                }
            })
            
        except Exception as e:
            logger.error(f"Error in receive_embeddings: {str(e)}")
            return Response({
                'code': 500,
                'msg': str(e),
                'data': None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['GET'])
    def latest_model(self, request):
        """Get information about the latest global model"""
        try:
            latest_model = GlobalModel.objects.latest('created_at')
            return Response({
                'code': 200,
                'msg': 'Latest model retrieved successfully',
                'data': {
                    'model_id': latest_model.id,
                    'version': latest_model.version,
                    'num_students': latest_model.num_students,
                    'created_at': latest_model.created_at
                }
            })
        except GlobalModel.DoesNotExist:
            return Response({
                'code': 404,
                'msg': 'No global model found',
                'data': None
            }, status=status.HTTP_404_NOT_FOUND)

    @action(detail=False, methods=['GET'])
    def check_model_files(self, request):
        """Debug endpoint to check model files"""
        try:
            # Check database
            models = GlobalModel.objects.all().order_by('-created_at')
            model_info = []
            
            for model in models:
                # Check file paths
                db_path = model.model_file.name
                full_path = os.path.join(settings.MEDIA_ROOT, db_path)
                alt_path = os.path.join(settings.MEDIA_ROOT, 'global_model', model.model_file)
                
                model_info.append({
                    'id': model.id,
                    'version': model.version,
                    'db_path': db_path,
                    'full_path_exists': os.path.exists(full_path),
                    'alt_path_exists': os.path.exists(alt_path),
                    'created_at': model.created_at
                })
            
            # Check global_model directory
            model_dir = os.path.join(settings.MEDIA_ROOT, 'global_model')
            files_in_dir = []
            if os.path.exists(model_dir):
                files_in_dir = os.listdir(model_dir)
            
            return Response({
                'code': 200,
                'msg': 'Model files check complete',
                'data': {
                    'models_in_db': model_info,
                    'files_in_directory': files_in_dir,
                    'media_root': settings.MEDIA_ROOT,
                    'model_dir': model_dir
                }
            })
            
        except Exception as e:
            return Response({
                'code': 500,
                'msg': str(e),
                'data': None
            })

    def draw_face_boxes(self, image, faces_info, match_results):
        """Draw boxes and labels on faces"""
        try:
            img = image.copy()
            
            for face, matches in zip(faces_info, [match_results]):
                x1, y1, x2, y2 = face['bbox']
                
                # Find valid matches
                valid_matches = [m for m in matches if m['similarity'] > 0.5]  # Using threshold of 0.5
                valid_matches.sort(key=lambda x: x['similarity'], reverse=True)
                
                if valid_matches:
                    # Only show the best match
                    match = valid_matches[0]  # Take only the best match
                    student_id = match['student_id']
                    similarity = match['similarity']
                    
                    # Extract last 3 digits
                    last_three_digits = student_id[-3:]
                    
                    # Different colors for different confidence levels
                    if similarity > 0.8:
                        color = (0, 255, 0)  # Green for high confidence
                    elif similarity > 0.65:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 165, 255)  # Orange for lower confidence
                    
                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw large roll number (last 3 digits)
                    font_scale = 2.0  # Much larger font size
                    font_thickness = 4  # Thicker font
                    
                    # Calculate text size for centering
                    (text_width, text_height), _ = cv2.getTextSize(
                        last_three_digits, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        font_thickness
                    )
                    
                    # Position the text above the box
                    text_x = x1 + (x2 - x1 - text_width) // 2  # Center horizontally
                    text_y = y1 - 20  # Position above the box
                    
                    # Draw the large roll number
                    cv2.putText(
                        img, 
                        last_three_digits,  # Only the last 3 digits
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        font_thickness
                    )
                    
                    # Draw smaller similarity score
                    similarity_text = f"({similarity:.2f})"
                    cv2.putText(
                        img,
                        similarity_text,
                        (text_x, text_y + text_height + 5),  # Position below the roll number
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,  # Smaller font for similarity
                        color,
                        1
                    )
                else:
                    # Red box for no match
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        img,
                        "No match",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
            
            return img
            
        except Exception as e:
            logger.error(f"Error drawing result: {str(e)}")
            return image

    def reaggregate_all_embeddings(self):
        """Reaggregate all available embeddings into a new global model"""
        try:
            # Get all student embeddings
            students = Student.objects.all()
            if not students:
                return None, "No student embeddings found"

            logger.info(f"Found {len(students)} students in database")

            # Load all embeddings
            embeddings_dict = {}
            for student in students:
                try:
                    # Get embedding file path
                    embedding_path = os.path.join(settings.MEDIA_ROOT, student.embedding_file.name)
                    logger.info(f"Loading embedding for {student.student_id} from {embedding_path}")
                    
                    # Load and normalize embedding
                    embedding = np.load(embedding_path)
                    embedding = embedding.astype(np.float32)
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings_dict[student.student_id] = embedding
                    
                    logger.info(f"Successfully loaded embedding for {student.student_id}")
                except Exception as e:
                    logger.error(f"Error loading embedding for student {student.student_id}: {str(e)}")
                    continue

            if not embeddings_dict:
                return None, "No valid embeddings found"

            logger.info(f"Successfully loaded {len(embeddings_dict)} embeddings")

            # Save new global model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"global_model_{timestamp}.npz"
            
            # Ensure directory exists
            os.makedirs(os.path.join(settings.MEDIA_ROOT, 'global_model'), exist_ok=True)
            
            # Save with full path
            model_path = os.path.join(settings.MEDIA_ROOT, 'global_model', model_filename)
            
            # Save numpy array with all embeddings
            np.savez(
                model_path,
                student_embeddings=embeddings_dict,
                num_students=len(embeddings_dict),
                timestamp=timestamp
            )
            
            logger.info(f"Saved model to {model_path}")
            
            # Create database entry with relative path
            relative_path = os.path.join('global_model', model_filename)
            global_model = GlobalModel.objects.create(
                version=timestamp,
                model_file=relative_path,
                num_students=len(embeddings_dict)
            )
            
            logger.info(f"Created database entry for model version {timestamp}")
            
            return global_model, None
            
        except Exception as e:
            logger.error(f"Error in reaggregate_all_embeddings: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, str(e)

    @action(detail=False, methods=['POST'])
    def reaggregate(self, request):
        """Endpoint to reaggregate all embeddings into a new model"""
        try:
            # Delete old model files
            model_dir = os.path.join(settings.MEDIA_ROOT, 'global_model')
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    file_path = os.path.join(model_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            # Reaggregate embeddings
            global_model, error = self.reaggregate_all_embeddings()
            
            if error:
                return Response({
                    'code': 400,
                    'msg': 'Error during reaggregation',
                    'data': {'error': error}
                }, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({
                'code': 200,
                'msg': 'Model reaggregated successfully',
                'data': {
                    'model_id': global_model.id,
                    'version': global_model.version,
                    'num_students': global_model.num_students,
                    'created_at': global_model.created_at
                }
            })
            
        except Exception as e:
            return Response({
                'code': 500,
                'msg': str(e),
                'data': None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['GET'])
    def check_embeddings(self, request):
        """Debug endpoint to check embeddings"""
        try:
            model_embeddings = self.load_latest_model()
            
            embedding_info = []
            for student_id, embedding in model_embeddings.items():
                embedding = np.array(embedding)
                norm = np.linalg.norm(embedding)
                embedding_info.append({
                    'student_id': student_id,
                    'shape': embedding.shape,
                    'norm': float(norm),
                    'min': float(np.min(embedding)),
                    'max': float(np.max(embedding)),
                    'mean': float(np.mean(embedding))
                })
            
            return Response({
                'code': 200,
                'msg': 'Embedding check complete',
                'data': {
                    'num_embeddings': len(model_embeddings),
                    'embeddings': embedding_info
                }
            })
            
        except Exception as e:
            return Response({
                'code': 500,
                'msg': str(e),
                'data': None
            })

    def base64_to_image(self, base64_string: str):
        """Convert base64 string to image"""
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            base64_string = base64_string.strip()
            img_bytes = base64.b64decode(base64_string)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None, "Failed to decode image"
            
            # Resize if too large
            max_size = 640
            if max(img.shape) > max_size:
                scale = max_size / max(img.shape)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None
            
        except Exception as e:
            return None, str(e)

    @action(detail=False, methods=['POST'], parser_classes=[JSONParser])
    def process_base64_images(self, request):
        """Process base64 images and generate embeddings"""
        try:
            # Get JSON data
            data = request.data
            if not isinstance(data, dict):
                return Response({
                    'code': 400,
                    'msg': 'Invalid request format',
                    'data': None
                }, status=status.HTTP_400_BAD_REQUEST)

            student_id = data.get('student_id')
            if not student_id:
                return Response({
                    'code': 400,
                    'msg': 'Missing student_id',
                    'data': None
                }, status=status.HTTP_400_BAD_REQUEST)

            images_data = data.get('images', [])

            if not images_data:
                return Response({
                    'code': 400,
                    'msg': 'Missing images data',
                    'data': None
                }, status=status.HTTP_400_BAD_REQUEST)

            # Process images and generate embeddings
            student_embeddings = []
            failed_images = []

            for idx, image_data in enumerate(images_data):
                # Get base64 string
                base64_string = image_data.get('base64_string')
                if not base64_string:
                    failed_images.append(idx)
                    continue

                # Convert to image
                img, error = self.base64_to_image(base64_string)
                if error:
                    self.logger.error(f"Error processing image {idx}: {error}")
                    failed_images.append(idx)
                    continue

                # Get face embedding
                faces = self.face_app.get(img)
                if not faces:
                    self.logger.warning(f"No face detected in image {idx}")
                    failed_images.append(idx)
                    continue

                # Get largest face
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                student_embeddings.append(face.embedding)

            if not student_embeddings:
                return Response({
                    'code': 400,
                    'msg': 'No valid embeddings generated',
                    'data': None
                }, status=status.HTTP_400_BAD_REQUEST)

            # Calculate average embedding
            avg_embedding = np.mean(student_embeddings, axis=0)
            normalized_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            # Save embedding file with roll number
            embeddings_dir = os.path.join(settings.MEDIA_ROOT, 'embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            embedding_filename = f"{student_id}_embedding.npy"
            embedding_path = os.path.join(embeddings_dir, embedding_filename)
            np.save(embedding_path, normalized_embedding)

            # Update student record with roll number
            Student.objects.update_or_create(
                student_id=student_id,
                defaults={'embedding_file': f'embeddings/{embedding_filename}'}
            )

            # Trigger reaggregation
            self.reaggregate(request)

            return Response({
                'code': 200,
                'msg': 'Embeddings generated and model updated successfully',
                'data': {
                    'student_id': student_id,
                    'processed_images': len(student_embeddings),
                    'failed_images': failed_images,
                    'embedding_file': f'embeddings/{embedding_filename}'
                }
            })

        except Exception as e:
            self.logger.error(f"Error in process_base64_images: {str(e)}")
            return Response({
                'code': 500,
                'msg': str(e),
                'data': None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
