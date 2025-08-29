import { Controller, Post, Get, Body, HttpCode, HttpStatus, UseGuards, Request } from '@nestjs/common';
import { AuthService } from './auth.service';
import { RegisterDto } from './dto/register.dto';
import { LoginDto } from './dto/login.dto';
import { JwtAuthGuard } from './guards/jwt-auth.guard';
import { 
  ApiTags, 
  ApiOperation, 
  ApiResponse as SwaggerResponse, 
  ApiBody,
  ApiBearerAuth, 
  ApiResponse
} from '@nestjs/swagger';
import { Public } from './decorators/public.decorator';

@ApiTags('Authentication')
@Controller('auth')
export class AuthController {
  constructor(private authService: AuthService) {}

  @Post('register')
  @ApiBody({ type: RegisterDto })
  @Public()
  @ApiOperation({ summary: 'Register a new user' })
  @SwaggerResponse({ 
    status: 201, 
    description: 'User successfully registered'
  })
  async register(@Body() registerDto: RegisterDto) {
    return this.authService.register(registerDto);
  }

  @Public()
  @Post('login')
  @ApiOperation({ summary: 'User login' })
  @ApiResponse({ 
    status: 200, 
    description: 'Login successful',
    schema: {
      example: {
        access_token: 'jwt_token_here',
        user: {
          id: 'user_id',
          rollNumber: 'roll_number',
          firstName: 'first_name',
          lastName: 'last_name',
          role: 'role'
        }
      }
    }
  })
  @ApiResponse({ status: 401, description: 'Unauthorized' })
  async login(@Body() loginDto: LoginDto) {
    return this.authService.login(loginDto);
  }

  @Get('profile/details')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Get student profile with marks and attendance' })
  @ApiResponse({ 
    status: 200, 
    description: 'Returns student profile with marks and attendance'
  })
  async getProfileDetails(@Request() req) {
    return this.authService.getProfileDetails(req.user.rollNumber);
  }

  @Get('students')
  // @UseGuards(JwtAuthGuard)
  // @ApiBearerAuth()
  @ApiOperation({ summary: 'Get all students' })
  @ApiResponse({ 
    status: 200, 
    description: 'Returns list of all students'
  })
  async getAllStudents() {
    return this.authService.getAllStudents();
  }
}
