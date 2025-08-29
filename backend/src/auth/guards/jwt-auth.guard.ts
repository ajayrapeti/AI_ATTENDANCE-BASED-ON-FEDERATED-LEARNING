import { Injectable, ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';

@Injectable()
export class JwtAuthGuard extends AuthGuard('jwt') {
  async canActivate(context: ExecutionContext): Promise<boolean> {
    // Call authenticate first
    const result = (await super.canActivate(context)) as boolean;
    const request = context.switchToHttp().getRequest();
    
    // Log authentication result and user
    console.log('JwtAuthGuard - Authentication result:', result);
    console.log('JwtAuthGuard - User:', request.user);
    
    return result;
  }

  handleRequest(err: any, user: any, info: any, context: ExecutionContext) {
    console.log('JwtAuthGuard - HandleRequest');
    console.log('Error:', err);
    console.log('User:', user);
    console.log('Info:', info);

    if (err || !user) {
      throw err || new UnauthorizedException('Invalid token or user not found');
    }

    // Get the request object and explicitly attach the user
    const request = context.switchToHttp().getRequest();
    request.user = user;
    
    console.log('JwtAuthGuard - User attached to request:', request.user);
    return user;
  }
} 