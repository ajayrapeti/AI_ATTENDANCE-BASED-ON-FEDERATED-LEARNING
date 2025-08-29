import { Injectable, Logger } from '@nestjs/common';

@Injectable()
export class CustomLogger extends Logger {
  logRequest(method: string, url: string, body?: any) {
    this.log(`${method} ${url}${body ? `\nBody: ${JSON.stringify(body, null, 2)}` : ''}`);
  }

  logResponse(method: string, url: string, response: any) {
    this.log(
      `Response for ${method} ${url}\n${JSON.stringify(response, null, 2)}`,
      'Response'
    );
  }

  logError(error: any, context?: string) {
    this.error(
      `Error Details:\n${JSON.stringify({
        message: error.message,
        stack: error.stack,
        ...(error.response && { response: error.response }),
      }, null, 2)}`,
      error.stack,
      context
    );
  }
} 