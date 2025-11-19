# 🚀 Task Manager REST API - Setup Guide

## 📋 Overview

This guide will walk you through setting up the **Task Manager REST API** built with **Nest.js** and **TypeScript**. The API provides CRUD operations for task management with JSON file storage.

## 🛠️ Technology Stack

- **Node.js** v24.11.1 (Latest LTS)
- **Nest.js** v11.1.9
- **TypeScript** v5.x
- **class-validator** for input validation
- **Jest** for testing
- **JSON** for data persistence

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

### Required Software
- **Node.js** v24.x LTS or higher
- **npm** v9.x or higher (comes with Node.js)
- **Git** for version control

### Verify Installation
```bash
# Check Node.js version
node --version
# Should output: v24.11.1 or higher

# Check npm version
npm --version
# Should output: 9.x or higher

# Check Git version
git --version
```

## 🚀 Installation Steps

### Step 1: Clone or Setup Project

**Option A: Clone from Repository (if available)**
```bash
git clone <repository-url>
cd task-manager-api
```

**Option B: Start from Scratch**
```bash
mkdir task-manager-api
cd task-manager-api
npm init -y
```

### Step 2: Install Nest.js CLI
```bash
# Install globally
npm install -g @nestjs/cli@latest

# Verify installation
nest --version
```

### Step 3: Install Dependencies

**Core Dependencies:**
```bash
npm install @nestjs/core@11.1.9
npm install @nestjs/common@11.0.11
npm install @nestjs/platform-express@11.0.12
npm install class-validator@0.14.1
npm install class-transformer@0.5.1
npm install @nestjs/config@3.3.0
npm install uuid@9.0.1

# Database Dependencies (Optional - for SQLite + Prisma)
npm install prisma@5.22.0 @prisma/client@5.22.0
```

**Development Dependencies:**
```bash
npm install -D @nestjs/cli@11.0.12
npm install -D @nestjs/schematics@11.0.12
npm install -D @nestjs/testing@11.0.12
npm install -D typescript@5.7.2
npm install -D @types/node@20.10.5
npm install -D @types/uuid@9.0.8
npm install -D ts-node@10.9.2
npm install -D jest@29.7.0
npm install -D @types/jest@29.5.14
npm install -D supertest@6.3.4
npm install -D @types/supertest@2.0.16
npm install -D eslint@8.56.0
npm install -D @nestjs/eslint-config-nestjs-tslint@0.0.1
npm install -D prettier@3.1.1
```

### Step 4: Create Project Structure

```bash
# Create directories
mkdir -p src/tasks/{dto,entities}
mkdir -p src/common/{filters,pipes}
mkdir -p data
mkdir -p test/tasks

# Create initial files (will be populated during implementation)
touch src/main.ts
touch src/app.module.ts
touch src/tasks/tasks.module.ts
touch src/tasks/tasks.controller.ts
touch src/tasks/tasks.service.ts
touch src/tasks/entities/task.entity.ts
touch src/tasks/dto/create-task.dto.ts
touch src/tasks/dto/update-task.dto.ts
touch data/tasks.json
```

### Step 5: Configuration Files

**Create `tsconfig.json`:**
```json
{
  "compilerOptions": {
    "module": "commonjs",
    "declaration": true,
    "removeComments": true,
    "emitDecoratorMetadata": true,
    "experimentalDecorators": true,
    "allowSyntheticDefaultImports": true,
    "target": "ES2020",
    "sourceMap": true,
    "outDir": "./dist",
    "baseUrl": "./",
    "incremental": true,
    "skipLibCheck": true,
    "strictNullChecks": false,
    "noImplicitAny": false,
    "strictBindCallApply": false,
    "forceConsistentCasingInFileNames": false,
    "noFallthroughCasesInSwitch": false
  }
}
```

**Create `nest-cli.json`:**
```json
{
  "$schema": "https://json.schemastore.org/nest-cli",
  "collection": "@nestjs/schematics",
  "sourceRoot": "src",
  "compilerOptions": {
    "deleteOutDir": true
  }
}
```

**Create `package.json` scripts:**
```json
{
  "scripts": {
    "build": "nest build",
    "format": "prettier --write \"src/**/*.ts\" \"test/**/*.ts\"",
    "start": "nest start",
    "start:dev": "nest start --watch",
    "start:debug": "nest start --debug --watch",
    "start:prod": "node dist/main",
    "lint": "eslint \"{src,apps,libs,test}/**/*.ts\" --fix",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:cov": "jest --coverage",
    "test:debug": "node --inspect-brk -r tsconfig-paths/register -r ts-node/register node_modules/.bin/jest --runInBand",
    "test:e2e": "jest --config ./test/jest-e2e.json"
  }
}
```

### Step 6: Prisma Database Setup (Optional)

**If you want to use SQLite + Prisma instead of JSON file storage:**

```bash
# Initialize Prisma with SQLite
npx prisma init --datasource-provider sqlite

# This creates:
# - prisma/schema.prisma (database schema file)
# - .env (environment variables with DATABASE_URL)
```

**Create Prisma Schema (`prisma/schema.prisma`):**
```prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "sqlite"
  url      = env("DATABASE_URL")
}

model Task {
  id          String    @id @default(cuid())
  title       String
  description String?
  status      TaskStatus @default(PENDING)
  createdAt   DateTime  @default(now()) @map("created_at")
  updatedAt   DateTime  @updatedAt @map("updated_at")

  @@map("tasks")
  @@index([status])
  @@index([createdAt])
  @@index([title])
}

enum TaskStatus {
  PENDING
  COMPLETED
}
```

**Generate Prisma Client:**
```bash
npx prisma generate
npx prisma migrate dev --name init
```

**Add Prisma Scripts to `package.json`:**
```json
{
  "scripts": {
    // ... existing scripts
    "db:generate": "prisma generate",
    "db:migrate": "prisma migrate dev",
    "db:studio": "prisma studio",
    "db:seed": "ts-node prisma/seed.ts",
    "db:reset": "prisma migrate reset"
  }
}
```

### Step 7: Environment Setup

**Create `.env` file:**
```env
# Server Configuration
PORT=3000
NODE_ENV=development

# Storage Configuration (Choose one)
USE_JSON_STORAGE=true
# DATABASE_URL="file:./dev.db"  # Uncomment for Prisma + SQLite

# File Storage (for JSON storage)
DATA_FILE_PATH=./data/tasks.json

# Database Configuration (for Prisma + SQLite)
# DATABASE_URL="file:./dev.db"

# Validation
MAX_TITLE_LENGTH=100
MAX_DESCRIPTION_LENGTH=500
```

**Create `.gitignore`:**
```
# Dependencies
node_modules/

# Build outputs
dist/
build/

# Environment files
.env
.env.local
.env.*.local

# Database files
*.db
*.db-journal
dev.db
prisma/dev.db*
prisma/migrations/

# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

## 🏃‍♂️ Running the Application

### Development Mode
```bash
# Install dependencies
npm install

# Start development server with hot reload
npm run start:dev

# The API will be available at http://localhost:3000
```

### Choosing Storage Backend

**Option 1: JSON File Storage (Default)**
```bash
# Use the default JSON file storage
# No additional setup needed
# Data is stored in ./data/tasks.json
```

**Option 2: SQLite + Prisma**
```bash
# Set up database storage
1. Follow Step 6 above to configure Prisma
2. Update .env file:
   USE_JSON_STORAGE=false
   DATABASE_URL="file:./dev.db"
3. Generate database schema:
   npm run db:migrate
4. Start the application
   npm run start:dev
```

### Database Management (Prisma Only)

```bash
# View and edit database in browser
npm run db:studio

# Reset database (delete all data)
npm run db:reset

# Generate new migration after schema changes
npm run db:migrate

# Regenerate Prisma Client
npm run db:generate
```

### Production Mode
```bash
# Build the application
npm run build

# Start production server
npm run start:prod
```

### Testing
```bash
# Run unit tests
npm run test

# Run tests with coverage
npm run test:cov

# Run tests in watch mode
npm run test:watch

# Run e2e tests
npm run test:e2e
```

## 🔧 API Endpoints

Once the server is running, you can access the following endpoints:

### Base URL
```
http://localhost:3000
```

### Available Endpoints
- `GET /tasks` - Get all tasks (with optional status filter)
- `GET /tasks/:id` - Get a specific task
- `POST /tasks` - Create a new task
- `PUT /tasks/:id` - Update a task
- `DELETE /tasks/:id` - Delete a task

### Query Parameters
- `?status=pending` - Filter tasks by status
- `?status=completed` - Filter completed tasks

## 📝 Development Workflow

### 1. Make Changes
Edit source files in the `src/` directory

### 2. Automatic Restart
The development server will automatically restart when files are saved

### 3. Run Tests
```bash
npm run test:watch
```

### 4. Linting
```bash
npm run lint
```

### 5. Formatting
```bash
npm run format
```

## 🐛 Troubleshooting

### Common Issues

**Issue: Port already in use**
```bash
# Find process using port 3000
lsof -ti:3000

# Kill the process
kill -9 <PID>
```

**Issue: TypeScript compilation errors**
```bash
# Clean build cache
rm -rf dist/
npm run build
```

**Issue: Permission denied**
```bash
# On macOS/Linux
sudo chown -R $USER:$USER .

# On Windows
# Run PowerShell as Administrator
```

**Issue: Module not found**
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

## 📚 Additional Resources

- [Nest.js Documentation](https://docs.nestjs.com/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Jest Testing Framework](https://jestjs.io/)
- [Express.js Documentation](https://expressjs.com/)

## 🆘 Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Ensure all dependencies are installed correctly
3. Verify Node.js version compatibility
4. Check file permissions
5. Review the [Troubleshooting](#-troubleshooting) section

## 🎯 Next Steps

After setup is complete:

1. Review the [ARCHITECTURE.md](./ARCHITECTURE.md) file
2. Start implementing the task management features
3. Run the provided tests
4. Extend the API with additional features
5. Deploy to production when ready

---

**Happy Coding!** 🎉