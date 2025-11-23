# OptoPomodoro Backend

A high-performance REST API and WebSocket server for the OptoPomodoro productivity application. Built with NestJS, TypeScript, and a focus on security, scalability, and developer experience.

## Overview

The OptoPomodoro backend provides a robust foundation for task management, Pomodoro tracking, team collaboration, and gamification features. Designed for Optomatica employees, it combines modern Node.js architecture with enterprise-grade security practices.

## Key Features

### 🔐 **Authentication & Security**
- JWT-based authentication with refresh token rotation
- Optomatica domain validation (@optomatica.com)
- Rate limiting and request throttling
- Password hashing with bcrypt (12+ salt rounds)
- CORS configuration for frontend domains

### 📊 **Task Management**
- Full CRUD operations for tasks
- Advanced filtering and sorting
- Priority levels (LOW, MEDIUM, HIGH, URGENT)
- Status tracking (TODO, IN_PROGRESS, COMPLETED, CANCELLED)
- Pomodoro estimation and completion tracking
- Team-based task assignment

### 🍅 **Session Management**
- Pomodoro timer session tracking
- Quality ratings and progress metrics
- Session analytics and insights
- Real-time timer synchronization
- Wellness and productivity scoring

### 👥 **Team Collaboration**
- Team creation and management
- Member roles and permissions
- Real-time updates via WebSocket
- Team challenges and competitions
- Shared task boards and goals

### 🎮 **Gamification System**
- XP calculation and level progression
- Achievement unlocking system
- Productivity streaks tracking
- Wellness score computation
- Leaderboards and team competitions

### 📡 **Real-time Features**
- WebSocket integration with Socket.IO
- Live task status updates
- Timer synchronization across devices
- Team presence indicators
- Real-time notifications

## Technology Stack

- **NestJS 11.x** - Modern Node.js framework with TypeScript
- **TypeScript 5.x** - Type safety and enhanced DX
- **Prisma 5.x** - Modern ORM with SQLite (PostgreSQL-ready)
- **JWT** - Stateless authentication with refresh tokens
- **Socket.IO 4.x** - Real-time bidirectional communication
- **bcryptjs** - Secure password hashing
- **Winston** - Structured logging and monitoring
- **Jest** - Comprehensive testing framework
- **Swagger** - API documentation and exploration

## Architecture

### Module Structure
```
src/
├── auth/                 # Authentication & authorization
├── users/                # User management and profiles
├── tasks/                # Task CRUD and Kanban operations
├── sessions/             # Pomodoro timer sessions
├── teams/                # Team collaboration features
├── gamification/         # XP, achievements, challenges
├── websocket/            # Real-time communication
├── common/               # Shared utilities and decorators
├── config/               # Configuration management
└── core/                 # Core services and middleware
```

### Database Schema
- **Users** - Authentication, preferences, gamification stats
- **Tasks** - Task management with priorities and Pomodoro tracking
- **Sessions** - Timer sessions with quality ratings
- **Teams** - Team structure and member management
- **Achievements** - Gamification system
- **Notifications** - User notifications and alerts

### API Design
- **RESTful endpoints** following OpenAPI 3.0 specification
- **Consistent response format** with proper error handling
- **Pagination and filtering** for list endpoints
- **Validation** using class-validator and class-transformer
- **Rate limiting** and security middleware

## Development

### Prerequisites
- Node.js 18+
- pnpm (preferred package manager)
- SQLite 3.x (development) or PostgreSQL 14+ (production)

### Setup
```bash
# Install dependencies
pnpm install

# Generate Prisma client
pnpm db:generate

# Run database migrations
pnpm db:migrate

# Seed database with test data (optional)
pnpm db:seed

# Start development server
pnpm dev
```

### Environment Variables
```bash
NODE_ENV=development
PORT=3001
DATABASE_URL="file:./dev.db"
JWT_SECRET=your-super-secret-jwt-key
JWT_REFRESH_SECRET=your-refresh-token-secret
FRONTEND_URL=http://localhost:3000
BCRYPT_ROUNDS=12
RATE_LIMIT_TTL=60
RATE_LIMIT_LIMIT=100
```

## Available Scripts

### Development
- `pnpm dev` - Start development server with hot reload
- `pnpm build` - Build the application for production
- `pnpm start:debug` - Start in debug mode with watch
- `pnpm start:prod` - Start production build

### Database
- `pnpm db:generate` - Generate Prisma client
- `pnpm db:push` - Push schema changes to database
- `pnpm db:migrate` - Run database migrations
- `pnpm db:seed` - Seed database with test data
- `pnpm db:studio` - Open Prisma Studio for database inspection

### Testing
- `pnpm test` - Run unit tests
- `pnpm test:watch` - Run tests in watch mode
- `pnpm test:cov` - Run tests with coverage report
- `pnpm test:e2e` - Run end-to-end tests
- `pnpm test:debug` - Run tests in debug mode

### Code Quality
- `pnpm lint` - Run ESLint
- `pnpm lint:fix` - Fix linting issues
- `pnpm format` - Format code with Prettier
- `pnpm type-check` - Run TypeScript type checking

## API Documentation

### Authentication
```http
POST /api/auth/register      # User registration
POST /api/auth/login         # User login
POST /api/auth/refresh       # Refresh token
POST /api/auth/logout        # User logout
```

### Tasks
```http
GET    /api/tasks           # Get user tasks with filters
POST   /api/tasks           # Create new task
GET    /api/tasks/:id       # Get specific task
PATCH  /api/tasks/:id       # Update task
DELETE /api/tasks/:id       # Delete task
```

### Sessions
```http
GET    /api/sessions        # Get user sessions
POST   /api/sessions        # Start timer session
PATCH  /api/sessions/:id    # Update session
DELETE /api/sessions/:id    # Delete session
```

### Teams
```http
GET    /api/teams           # Get user teams
POST   /api/teams           # Create team
GET    /api/teams/:id       # Get specific team
PATCH  /api/teams/:id       # Update team
DELETE /api/teams/:id       # Delete team
```

### WebSocket Events
```javascript
// Client-side
socket.emit('timer:start', { taskId, duration });
socket.emit('timer:complete', { sessionId, quality });
socket.emit('task:update', { taskId, status });

// Server-side
socket.on('timer:tick', (remainingTime) => {});
socket.on('task:updated', (task) => {});
socket.on('team:update', (teamData) => {});
```

## Security Features

### Authentication
- **JWT Access Tokens**: 15-minute expiration
- **Refresh Tokens**: 7-day expiration with rotation
- **Password Security**: bcrypt with 12+ salt rounds
- **Domain Validation**: Only @optomatica.com emails allowed

### API Security
- **Rate Limiting**: Configurable request limits
- **CORS**: Properly configured for frontend domains
- **Input Validation**: Comprehensive DTO validation
- **SQL Injection Prevention**: Prisma ORM protection
- **XSS Protection**: Input sanitization and output encoding

### Infrastructure
- **Environment Variables**: Secure configuration management
- **Error Logging**: Winston with structured logs
- **Request Tracing**: Request ID tracking for debugging
- **Health Checks**: Built-in health monitoring endpoints

## Performance Optimizations

### Database
- **Connection Pooling**: Prisma connection management
- **Query Optimization**: Selective field loading
- **Indexing Strategy**: Optimized for common queries
- **Caching**: Response caching for frequently accessed data

### API
- **Response Compression**: Gzip compression middleware
- **Pagination**: Efficient data loading for large datasets
- **Lazy Loading**: Related data loaded on demand
- **Rate Limiting**: Prevents API abuse and overload

### Real-time
- **WebSocket Optimization**: Efficient event broadcasting
- **Connection Management**: Proper cleanup and error handling
- **Message Queuing**: Redis-ready for scaling

## Testing Strategy

### Unit Tests
- **Service Layer**: Business logic testing
- **Controller Layer**: API endpoint testing
- **Utility Functions**: Helper function testing
- **Mock Dependencies**: Isolated unit testing

### Integration Tests
- **API Endpoints**: Full request-response cycles
- **Database Operations**: Prisma integration testing
- **Authentication**: Auth flow testing
- **WebSocket Events**: Real-time feature testing

### E2E Tests
- **User Workflows**: Complete user journey testing
- **Cross-browser Compatibility**: Multiple browser testing
- **Mobile Compatibility**: Responsive design testing
- **Performance Testing**: Load and stress testing

## Production Deployment

### Environment Setup
- **Node.js 18+ LTS**: Stable runtime environment
- **PostgreSQL 14+**: Production database
- **Redis 6+**: Session storage and caching
- **PM2**: Process management and monitoring

### Docker Support
```dockerfile
# Multi-stage build for optimized image size
FROM node:18-alpine AS builder
# Build application

FROM node:18-alpine AS production
# Production runtime
```

### Monitoring
- **Health Endpoints**: `/health`, `/ready` for load balancers
- **Metrics**: Custom metrics for performance monitoring
- **Logging**: Structured JSON logs for log aggregation
- **Error Tracking**: Integration with error monitoring services

## Contributing

### Code Standards
- **TypeScript**: Strict mode enabled
- **ESLint**: Configured for NestJS best practices
- **Prettier**: Consistent code formatting
- **Husky**: Pre-commit hooks for code quality

### Development Workflow
1. **Fork** the repository
2. **Create feature branch** from develop
3. **Write tests** for new functionality
4. **Ensure all tests pass**
5. **Update documentation** as needed
6. **Submit pull request** with detailed description

### Git Commit Convention
```
feat: Add team challenge system
fix: Resolve authentication token refresh issue
docs: Update API documentation
refactor: Optimize database queries
test: Add integration tests for task creation
```

## License

© 2024 Optomatica. All rights reserved.

---

For detailed API documentation, visit: http://localhost:3001/api/docs (when running locally)