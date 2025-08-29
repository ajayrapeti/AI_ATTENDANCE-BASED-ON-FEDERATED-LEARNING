import { Injectable, UnauthorizedException } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { ExtractJwt, Strategy } from 'passport-jwt';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor() {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: process.env.JWT_SECRET,
    });
  }

  async validate(payload: any) {
    console.log('JwtStrategy - Validating payload:', payload);

    if (!payload) {
      throw new UnauthorizedException('Invalid token payload');
    }

    const user = {
      userId: payload.sub,
      rollNumber: payload.rollNumber,
      role: payload.role,
      firstName: payload.firstName,
      lastName: payload.lastName,
      subject: payload.subject
    };

    console.log('JwtStrategy - Created user object:', user);

    if (!user.role) {
      throw new UnauthorizedException('User role not found in token');
    }

    return user;
  }
} 