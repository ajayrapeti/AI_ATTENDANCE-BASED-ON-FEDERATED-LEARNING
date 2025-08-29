import { Injectable, CanActivate, ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { Role } from '../enums/role.enum';
import { ROLES_KEY } from '../decorators/roles.decorator';

@Injectable()
export class RolesGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  canActivate(context: ExecutionContext): boolean {
    console.log('RolesGuard - Starting role check');

    const requiredRoles = this.reflector.getAllAndOverride<Role[]>(ROLES_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);

    console.log('RolesGuard - Required roles:', requiredRoles);

    if (!requiredRoles) {
      return true;
    }

    const request = context.switchToHttp().getRequest();
    console.log('RolesGuard - Request headers:', request.headers);
    console.log('RolesGuard - Request user:', request.user);

    if (!request.user) {
      console.log('RolesGuard - No user found in request');
      throw new UnauthorizedException('User not found in request. Make sure JwtAuthGuard is applied first.');
    }

    if (!request.user.role) {
      console.log('RolesGuard - No role found in user object');
      throw new UnauthorizedException('User role not found');
    }

    // Convert roles to lowercase for comparison
    const userRole = request.user.role.toLowerCase();
    const hasRole = requiredRoles.some((role) => role.toLowerCase() === userRole);
    
    console.log('RolesGuard - User role:', userRole);
    console.log('RolesGuard - Required roles (lowercase):', requiredRoles.map(r => r.toLowerCase()));
    console.log('RolesGuard - Has required role:', hasRole);

    return hasRole;
  }
}