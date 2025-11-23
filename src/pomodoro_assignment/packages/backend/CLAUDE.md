# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OptoPomodoro** is a Zen-inspired productivity application for Optomatica employees. This is a monorepo containing both frontend and backend packages built with modern TypeScript technologies.

### Architecture
- **Frontend**: React 18 + TypeScript + Vite + PWA (Progressive Web App)
- **Backend**: NestJS + TypeScript + SQLite + Prisma ORM
- **Real-time**: Socket.IO for timer synchronization and team features
- **Authentication**: Password-first JWT with Optomatica domain validation (@optomatica.com)
- **Gamification**: XP system, achievements, streaks, and team challenges
- **Database**: SQLite with Prisma ORM (PostgreSQL-ready for production)

## Common Development Commands

### Backend (Current Directory)
```bash
# Development
pnpm dev                    # Start backend in development mode
nest start                 # Start in production mode
nest start --watch         # Start with file watching

# Building
pnpm build                 # Build the backend application
nest build                # Alternative build command

# Testing
pnpm test                  # Run unit tests
pnpm test:watch           # Run tests in watch mode
pnpm test:cov             # Run tests with coverage
pnpm test:e2e             # Run end-to-end tests
pnpm test:debug           # Run tests in debug mode

# Code Quality
pnpm lint                  # Run ESLint
pnpm format                # Format code with Prettier

# Database Operations
pnpm prisma:generate       # Generate Prisma client
pnpm prisma:migrate        # Run database migrations
pnpm prisma:studio         # Open Prisma Studio
pnpm prisma:seed          # Seed database with test data
```

### Frontend (From Root)
```bash
# Navigate to frontend first
cd ../frontend

# Development
pnpm dev                   # Start frontend development server
pnpm build                # Build for production
pnpm preview              # Preview production build

# Testing
pnpm test                 # Run unit tests with Vitest
pnpm test:e2e            # Run E2E tests with Cypress
pnpm test:e2e:open       # Open Cypress UI

# Code Quality
pnpm lint                 # Run ESLint
pnpm lint:fix            # Fix linting issues
pnpm type-check          # Run TypeScript type checking
```

### Root Level (Monorepo)
```bash
# From project root (one level up)
pnpm dev                  # Start both frontend and backend
pnpm build               # Build all packages
pnpm test                # Run all tests
pnpm lint                # Lint all packages
```

## Key Architecture Patterns

### 1. Modular NestJS Structure
The backend follows NestJS's modular architecture with dedicated modules for each domain:
- `auth/` - Authentication and authorization
- `users/` - User management and profiles
- `tasks/` - Task management and Kanban board
- `sessions/` - Pomodoro timer sessions
- `teams/` - Team collaboration features
- `gamification/` - Achievements, XP, and challenges
- `websocket/` - Real-time communication

### 2. Database Schema with Prisma
The database uses Prisma ORM with a schema that supports:
- **User accounts** with Optomatica email validation
- **Task management** with priority, status, and dependencies
- **Session tracking** for Pomodoro timers with quality ratings
- **Team collaboration** with roles and permissions
- **Achievement system** with XP calculation and streaks
- **Real-time features** through WebSocket integration

### 3. Authentication Flow
- **Password-first JWT authentication** (primary method)
- **Optomatica domain validation** (@optomatica.com emails only)
- **Refresh token rotation** for enhanced security
- **Role-based access control** for team features
- **OAuth support** ready for enterprise SSO (future enhancement)

### 4. Real-time Communication
- **Socket.IO integration** for timer synchronization
- **Team collaboration** with live updates
- **Achievement notifications** and progress tracking
- **Presence awareness** for team members

### 5. API Design Patterns
- **RESTful endpoints** with consistent `/api` prefix
- **DTO validation** using class-validator and class-transformer
- **Error handling** with structured error responses
- **Swagger documentation** at `/api/docs`
- **Rate limiting** and security middleware

## Development Workflow

### 1. Setting Up the Backend
```bash
# Install dependencies
pnpm install

# Generate Prisma client
pnpm prisma:generate

# Run database migrations
pnpm prisma:migrate

# Seed database (optional)
pnpm prisma:seed

# Start development server
pnpm dev
```

### 2. Working with Database Schema
```bash
# Modify schema in prisma/schema.prisma
# Generate migration
pnpm prisma:migrate --name migration_name

# Regenerate Prisma client
pnpm prisma:generate

# Open database to inspect
pnpm prisma:studio
```

### 3. Testing Strategy
- **Unit tests**: Test individual services and utilities
- **Integration tests**: Test API endpoints with test database
- **E2E tests**: Test complete user workflows
- **Coverage goal**: 80%+ maintained across all modules

### 4. Code Standards
- **TypeScript strict mode** enabled
- **ESLint + Prettier** for consistent formatting
- **Husky pre-commit hooks** for code quality
- **Conventional commits** for change tracking

## Frontend Integration Points

### API Endpoints Structure
- `POST /api/auth/login` - User authentication
- `POST /api/auth/register` - User registration
- `GET /api/users/profile` - User profile data
- `GET /api/tasks` - Task list with filters
- `POST /api/tasks` - Create new task
- `POST /api/sessions` - Start timer session
- `PUT /api/sessions/:id/complete` - Complete session

### WebSocket Events
- `timer:start` - Begin Pomodoro session
- `timer:complete` - Finish session with quality rating
- `task:update` - Real-time task updates
- `user:online/offline` - Presence tracking

### Data Models Alignment
The backend schema is aligned with frontend TypeScript interfaces:
- **User**: `firstName`, `lastName`, `email`, `level`, `xp`, `streak`
- **Task**: `title`, `priority`, `status`, `estimatedPomodoros`, `tags`
- **Session**: `startTime`, `endTime`, `duration`, `quality`, `completed`

## Important Implementation Notes

### 1. Security Requirements
- **Optomatica email validation** is mandatory for all users
- **Password hashing** with bcrypt (12+ salt rounds)
- **JWT tokens** with 15-minute access, 7-day refresh
- **Rate limiting** on authentication endpoints
- **CORS configured** for frontend domains

### 2. Performance Considerations
- **Database indexing** on frequently queried fields
- **Prisma query optimization** with selective includes
- **Caching strategy** for user sessions and achievements
- **Connection pooling** for database operations

### 3. Error Handling
- **Global exception filter** for consistent error responses
- **Structured logging** with Winston for production
- **Validation errors** with detailed field-level feedback
- **Graceful degradation** for WebSocket disconnections

### 4. Testing Requirements
- **Test database** isolated from development data
- **Mock external dependencies** in unit tests
- **Integration tests** cover all major API endpoints
- **WebSocket testing** for real-time features

## Development Environment

### Environment Variables
Required `.env` file (see `.env.example`):
```
NODE_ENV=development
PORT=3001
DATABASE_URL=file:./dev.db
JWT_SECRET=your-super-secret-jwt-key
FRONTEND_URL=http://localhost:3000
```

### Database Setup
- **Development**: SQLite database (`./dev.db`)
- **Testing**: In-memory SQLite for isolation
- **Production**: PostgreSQL (configured via DATABASE_URL)

### API Documentation
- **Swagger UI**: Available at `http://localhost:3001/api/docs`
- **API prefix**: All endpoints under `/api`
- **Authentication**: Bearer token required for protected routes

## Common Issues and Solutions

### 1. Prisma Client Generation
If you get "cannot find module '@prisma/client'":
```bash
pnpm prisma:generate
```

### 2. Database Connection Issues
Ensure DATABASE_URL is correct and database file has proper permissions.

### 3. JWT Token Issues
Check JWT_SECRET is set and tokens are properly signed/verified.

### 4. CORS Issues
Verify FRONTEND_URL is in CORS allowed origins in main.ts.

## Future Enhancement Areas

The architecture is designed to support:
- **OAuth providers** (Google, Microsoft) for enterprise SSO
- **Redis caching** for session management and leaderboards
- **PostgreSQL migration** for production scalability
- **Microservices extraction** for specific domains
- **Analytics pipeline** for productivity insights
- **Mobile API** for native app development

This backend provides a solid foundation for the OptoPomodoro application with secure authentication, real-time collaboration, comprehensive gamification, and room for future growth.