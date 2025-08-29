import { Controller, Post, Body, HttpException, HttpStatus } from '@nestjs/common';
import { FaceEmbeddingRequest, FaceEmbeddingResponse } from './interfaces/face.interface';
import { TestService } from './test.service';

@Controller('test')
export class TestController {
  constructor(private readonly testService: TestService) {}

  @Post('face-embeddings')
  async saveFaceEmbeddings(
    @Body() faceEmbeddingRequest: FaceEmbeddingRequest
  ): Promise<FaceEmbeddingResponse> {
    try {
      const { studentId, embeddings } = faceEmbeddingRequest;

      // Validate request
      if (!studentId || !embeddings || !Array.isArray(embeddings)) {
        throw new HttpException(
          'Invalid request format',
          HttpStatus.BAD_REQUEST
        );
      }

      // Validate embeddings format
      const isValid = await this.testService.validateEmbeddings(embeddings);
      if (!isValid) {
        throw new HttpException(
          'Invalid embeddings format',
          HttpStatus.BAD_REQUEST
        );
      }

      // Save embeddings
      await this.testService.saveFaceEmbeddings(faceEmbeddingRequest);

      return {
        success: true,
        message: 'Face embeddings saved successfully',
        data: {
          savedEmbeddings: embeddings.length,
          studentId
        }
      };
    } catch (error) {
      console.error('Error in saveEmbeddings controller:', error);
      throw new HttpException(
        error.message || 'Internal server error',
        error.status || HttpStatus.INTERNAL_SERVER_ERROR
      );
    }
  }
} 