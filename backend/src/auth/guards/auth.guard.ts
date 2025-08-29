import { Injectable, ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { AuthGuard } from '@nestjs/passport';
import { Role } from '../enums/role.enum';
import { ROLES_KEY } from '../decorators/roles.decorator';
import { IS_PUBLIC_KEY } from '../decorators/public.decorator';

@Injectable()
export class AuthRolesGuard extends AuthGuard('jwt') {
  constructor(private reflector: Reflector) {
    super();
  }

  async canActivate(context: ExecutionContext): Promise<boolean> {
    // Check if route is marked as public
    const isPublic = this.reflector.getAllAndOverride<boolean>(IS_PUBLIC_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);

    if (isPublic) {
      return true;
    }

    // Rest of your existing guard logic
    console.log('AuthRolesGuard - Starting authentication and role check');

    try {
      const authenticated = await super.canActivate(context);
      if (!authenticated) {
        return false;
      }
    } catch (err) {
      throw new UnauthorizedException('Invalid token');
    }

    const request = context.switchToHttp().getRequest();
    
    const requiredRoles = this.reflector.getAllAndOverride<Role[]>(ROLES_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);

    console.log('AuthRolesGuard - Required roles:', requiredRoles);

    if (!requiredRoles) {
      return true;
    }

    const user = await this.handleRequest(null, request.user, null);

    if (!user) {
      console.log('AuthRolesGuard - No user found in request after authentication');
      throw new UnauthorizedException('User not found in request');
    }

    if (!user.role) {
      console.log('AuthRolesGuard - No role found in user object');
      throw new UnauthorizedException('User role not found');
    }

    const userRole = user.role.toLowerCase();
    const hasRole = requiredRoles.some((role) => role.toLowerCase() === userRole);

    console.log('AuthRolesGuard - User role:', userRole);
    console.log('AuthRolesGuard - Has required role:', hasRole);

    return hasRole;
  }

  handleRequest(err: any, user: any, info: any) {
    console.log('AuthRolesGuard - Handling request');
    console.log('Error:', err);
    console.log('User:', user);
    console.log('Info:', info);

    if (err || !user) {
      throw err || new UnauthorizedException('Invalid token or user not found');
    }

    return user;
  }
} 