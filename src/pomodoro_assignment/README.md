# OptoPomodoro

A Zen-inspired productivity application that combines the Pomodoro Technique with mindfulness principles, built for Optomatica teams.

## 🌟 Overview

OptoPomodoro is a modern task management application designed to help teams achieve focused productivity while maintaining mental wellness. Built with a monorepo architecture using Turborepo, it provides a seamless experience across web and mobile platforms.

### Key Philosophy

- **Focus Over Speed**: Quality work over rushing through tasks
- **Mindful Productivity**: Regular breaks and wellness reminders
- **Team Collaboration**: Work together while maintaining individual focus
- **Gamified Motivation**: Achievements, streaks, and team challenges
- **Zen Interface**: Clean, calming design that reduces distractions

## 🏗️ Monorepo Architecture

This project uses **Turborepo** to manage multiple packages with shared tooling and optimized build pipelines:

```
optopomodoro/
├── packages/
│   ├── frontend/          # React + TypeScript + Vite
│   └── backend/           # NestJS + TypeScript + Prisma
├── turbo.json            # Turborepo configuration
├── pnpm-workspace.yaml   # PNPM workspace config
└── package.json          # Root package scripts
```

### Package Overview

#### 📱 Frontend Package (`packages/frontend/`)
- **React 18** with TypeScript and modern hooks
- **Vite** for lightning-fast development and builds
- **Redux Toolkit** for state management with RTK Query
- **Styled Components** with design system
- **React DnD** for drag-and-drop Kanban interface
- **Framer Motion** for smooth animations
- **PWA** capabilities for mobile experience

#### 🔧 Backend Package (`packages/backend/`)
- **NestJS** with modern TypeScript patterns
- **Prisma ORM** with SQLite (PostgreSQL-ready)
- **JWT authentication** with refresh token rotation
- **Socket.IO** for real-time collaboration
- **Winston logging** and comprehensive error handling
- **OpenAPI/Swagger** documentation

## 🚀 Quick Start

### Prerequisites
- **Node.js 18+** with npm/pnpm
- **Git** for version control

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd optopomodoro

# Install dependencies (uses pnpm workspaces)
pnpm install

# Set up environment variables
cp packages/backend/.env.example packages/backend/.env
cp packages/frontend/.env.example packages/frontend/.env

# Generate Prisma client and run migrations
pnpm db:generate
pnpm db:migrate

# Seed database with test data (optional)
pnpm db:seed
```

### Development
```bash
# Start both frontend and backend in development mode
pnpm dev

# Or run individually:
pnpm --filter frontend dev  # Frontend: http://localhost:3000
pnpm --filter backend dev   # Backend:  http://localhost:3001
```

## 📋 Available Scripts

### Development Commands
```bash
pnpm dev              # Start all packages in development
pnpm build            # Build all packages
pnpm test             # Run all tests
pnpm lint             # Lint all packages
pnpm lint:fix         # Fix linting issues
pnpm type-check       # Type check all packages
```

### Database Commands
```bash
pnpm db:generate      # Generate Prisma client
pnpm db:push         # Push schema to database
pnpm db:migrate      # Run database migrations
pnpm db:seed         # Seed database with test data
pnpm db:studio       # Open Prisma Studio
```

### Package-Specific Commands
```bash
# Frontend only
pnpm --filter frontend dev
pnpm --filter frontend build
pnpm --filter frontend test

# Backend only
pnpm --filter backend dev
pnpm --filter backend build
pnpm --filter backend test
```

## 🛠️ Technology Stack

### Monorepo Management
- **Turborepo 2.x** - Build system and task orchestration
- **PNPM 9.x** - Fast, disk space efficient package manager
- **Husky** - Git hooks for code quality
- **lint-staged** - Run linters on staged files

### Frontend Stack
- **React 18** - Modern UI with concurrent features
- **TypeScript 5.x** - Type safety and enhanced DX
- **Vite** - Lightning-fast build tool
- **Redux Toolkit** - State management with RTK Query
- **Styled Components** - CSS-in-JS with theming
- **React DnD** - Drag and drop functionality
- **Framer Motion** - Smooth animations
- **Socket.IO Client** - Real-time communication

### Backend Stack
- **NestJS 11.x** - Modern Node.js framework
- **TypeScript 5.x** - Type-safe backend development
- **Prisma 5.x** - Modern ORM with type safety
- **SQLite** - Development database (PostgreSQL for production)
- **JWT** - Authentication with refresh tokens
- **Socket.IO** - Real-time bidirectional communication
- **Winston** - Structured logging
- **Jest** - Testing framework

## 🔧 Configuration

### Turborepo Configuration
The `turbo.json` file defines task dependencies and caching:

```json
{
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}
```

### Workspace Configuration
`pnpm-workspace.yaml` defines the workspace structure:

```yaml
packages:
  - 'packages/*'
```

### Environment Variables

#### Backend (`packages/backend/.env`)
```bash
NODE_ENV=development
PORT=3001
DATABASE_URL="file:./dev.db"
JWT_SECRET=your-super-secret-jwt-key
FRONTEND_URL=http://localhost:3000
```

#### Frontend (`packages/frontend/.env`)
```bash
VITE_API_URL=http://localhost:3001
VITE_WS_URL=ws://localhost:3001
VITE_APP_NAME=OptoPomodoro
```

## 📦 Package Scripts

### Root Level Scripts
- **`dev`**: Start all packages in development mode
- **`build`**: Build all packages with dependency resolution
- **`test`**: Run all tests with proper dependency ordering
- **`lint`**: Lint all packages with consistent configuration
- **`clean`**: Clean all build artifacts and node_modules

### Package Dependencies
Turborepo automatically handles task dependencies:

- `build` tasks run after dependencies are built
- `test` tasks run after build completion
- `dev` tasks run in parallel with caching disabled

## 🧪 Testing Strategy

### Test Types
```bash
# Unit Tests
pnpm test                    # Run all unit tests
pnpm --filter backend test   # Backend unit tests
pnpm --filter frontend test  # Frontend unit tests

# E2E Tests
pnpm test:e2e               # Run all end-to-end tests

# Coverage
pnpm test --coverage        # Run tests with coverage
```

### Test Configuration
- **Jest** configuration at package level
- **Shared test utilities** in workspace
- **Test databases** isolated per package
- **CI/CD integration** with proper test coverage

## 🚀 Deployment

### Production Build
```bash
# Build all packages for production
pnpm build

# Test production build
pnpm test
pnpm type-check
```

### Docker Support
```bash
# Development environment
pnpm docker:dev

# Production build
pnpm docker:build
```

### Environment Setup
- **Development**: Local SQLite database
- **Staging**: Docker containers with PostgreSQL
- **Production**: Cloud deployment with scaling

## 🔍 Performance Optimizations

### Turborepo Benefits
- **Caching**: Intelligent task caching
- **Parallel Execution**: Run independent tasks in parallel
- **Incremental Builds**: Only rebuild changed packages
- **Dependency Resolution**: Automatic task ordering

### Frontend Optimizations
- **Code Splitting**: Automatic with Vite
- **Tree Shaking**: Remove unused code
- **Asset Optimization**: Image and font optimization
- **Service Worker**: PWA caching strategies

### Backend Optimizations
- **Connection Pooling**: Database connection management
- **Query Optimization**: Efficient Prisma queries
- **Response Caching**: API response caching
- **Compression**: Gzip response compression

## 📚 Documentation

### Package-Specific Documentation
- **Frontend**: `packages/frontend/README.md`
- **Backend**: `packages/backend/README.md`
- **API Documentation**: Available at `http://localhost:3001/api/docs`

### Code Documentation
- **JSDoc**: Comprehensive code documentation
- **Type Comments**: Enhanced TypeScript documentation
- **API Docs**: Auto-generated OpenAPI/Swagger documentation

## 🤝 Contributing

### Development Workflow
1. **Fork** the repository
2. **Create feature branch** from main
3. **Install dependencies**: `pnpm install`
4. **Make changes** to appropriate package
5. **Run tests**: `pnpm test`
6. **Check types**: `pnpm type-check`
7. **Lint code**: `pnpm lint:fix`
8. **Commit changes** with conventional commits
9. **Push branch** and create pull request

### Code Quality
- **TypeScript Strict Mode**: All packages use strict TypeScript
- **ESLint Configuration**: Consistent linting across packages
- **Prettier Formatting**: Consistent code formatting
- **Pre-commit Hooks**: Automatic code quality checks
- **Test Coverage**: Comprehensive test coverage requirements

### Git Commit Convention
```
feat: Add team challenge system
fix: Resolve authentication token refresh issue
docs: Update API documentation
refactor: Optimize database queries
test: Add integration tests for task creation
chore: Update dependencies
```

## 📄 License

© 2024 Optomatica. All rights reserved.

---

## 🌟 Acknowledgments

Built with ❤️ for Optomatica teams using modern web technologies and best practices.

**Built with Turborepo** for optimal developer experience and build performance.