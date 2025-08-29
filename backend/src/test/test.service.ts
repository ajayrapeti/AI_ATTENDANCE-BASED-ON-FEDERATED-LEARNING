import { Injectable } from '@nestjs/common';
import { FaceEmbedding, FaceEmbeddingRequest } from './interfaces/face.interface';
import * as fs from 'fs/promises';
import * as path from 'path';

@Injectable()
export class TestService {
  async saveFaceEmbeddings(data: FaceEmbeddingRequest): Promise<boolean> {
    try {
      console.log('Saving face embeddings for student:', data.studentId);
      console.log('Number of embeddings:', data.embeddings.length);

      // Save directly in the test folder
      const filePath = path.join(__dirname, 'embeddings.json');

      // Read existing data if file exists
      let existingData = [];
      try {
        const fileContent = await fs.readFile(filePath, 'utf8');
        existingData = JSON.parse(fileContent);
      } catch (error) {
        // File doesn't exist yet, start with empty array
        console.log('Creating new embeddings file');
      }

      // Add new embedding data
      existingData.push({
        studentId: data.studentId,
        timestamp: new Date().toISOString(),
        embeddings: data.embeddings
      });

      // Save to file
      await fs.writeFile(
        filePath,
        JSON.stringify(existingData, null, 2),
        'utf8'
      );

      console.log('Embeddings saved to file:', filePath);
      return true;
    } catch (error) {
      console.error('Error saving face embeddings:', error);
      throw error;
    }
  }

  async validateEmbeddings(embeddings: FaceEmbedding[]): Promise<boolean> {
    // Validate that embeddings are in the correct format
    return embeddings.every(embedding => 
      embedding.photoIndex >= 0 && 
      Array.isArray(embedding.embedding) && 
      embedding.embedding.length === 128 && // FaceAPI embeddings are 128-dimensional
      embedding.embedding.every(value => typeof value === 'number')
    );
  }
} 