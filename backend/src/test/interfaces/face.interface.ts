export interface FaceEmbedding {
  photoIndex: number;
  embedding: number[];
}

export interface FaceEmbeddingRequest {
  studentId: string;
  embeddings: FaceEmbedding[];
}

export interface FaceEmbeddingResponse {
  success: boolean;
  message: string;
  data?: {
    savedEmbeddings: number;
    studentId: string;
  };
} 