import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
  HttpStatus,
} from '@nestjs/common';
import { Observable } from 'rxjs';
import { map, tap } from 'rxjs/operators';
import { ApiResponse } from '../interfaces/api-response.interface';
import { CustomLogger } from '../logger/logger.service';

@Injectable()
export class ResponseInterceptor<T> implements NestInterceptor<T, ApiResponse<T>> {
  private readonly logger = new CustomLogger('ResponseInterceptor');

  intercept(
    context: ExecutionContext,
    next: CallHandler,
  ): Observable<ApiResponse<T>> {
    const ctx = context.switchToHttp();
    const request = ctx.getRequest();
    
    // Log the incoming request
    this.logger.logRequest(
      request.method,
      request.url,
      request.body
    );

    return next.handle().pipe(
      map(data => {
        const response = {
          code: context.switchToHttp().getResponse().statusCode || HttpStatus.OK,
          msg: 'Success',
          data: data || null,
        };

        // Log the response
        this.logger.logResponse(
          request.method,
          request.url,
          response
        );

        return response;
      })
    );
  }
} 