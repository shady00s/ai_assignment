# 🏗️ Task Manager REST API - Architecture Documentation

## 📋 Overview

This document outlines the architectural decisions, patterns, and design principles for the **Task Manager REST API** built with **Nest.js v11.1.9** and **TypeScript v5.x**. The architecture follows enterprise-grade patterns while maintaining simplicity for educational purposes.

## 🎯 Architectural Goals

1. **Simplicity & Learnability**: Easy for beginners to understand and modify
2. **Scalability**: Designed to accommodate future growth
3. **Maintainability**: Clean, organized code structure
4. **Testability**: Comprehensive test coverage
5. **Performance**: Efficient data handling and response times
6. **Type Safety**: Full TypeScript implementation

## 🏛️ Architecture Pattern: Layered Modular Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATION                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST API
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    GATEWAY LAYER                           │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   CONTROLLERS   │  │   MIDDLEWARE    │                  │
│  │   (HTTP Layer)  │  │   (Validation, │                  │
│  │                 │  │   Error Filter) │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────┬───────────────────────────────────────┘
                      │ Dependency Injection
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    BUSINESS LAYER                          │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │    SERVICES     │  │      DTOs       │                  │
│  │ (Logic Layer)   │  │  (Data Transfer │                  │
│  │                 │  │   Objects)      │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────┬───────────────────────────────────────┘
                      │ Data Access Interface
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER                       │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   TASKSMODEL    │  │   REPOSITORY    │                  │
│  │ (Data Model +   │  │ (File Storage   │                  │
│  │  Validation)    │  │  Implementation)│                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────┬───────────────────────────────────────┘
                      │ File Operations
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 PERSISTENCE LAYER                          │
│                   JSON FILE                               │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
task-manager-api/
├── 📂 src/                          # Source code
│   ├── 📂 tasks/                    # Tasks feature module
│   │   ├── 📂 dto/                  # Data Transfer Objects
│   │   │   ├── 📄 create-task.dto.ts
│   │   │   ├── 📄 update-task.dto.ts
│   │   │   └── 📄 query-tasks.dto.ts
│   │   ├── 📂 entities/             # Data models
│   │   │   └── 📄 task.entity.ts
│   │   ├── 📂 models/               # Data access layer
│   │   │   ├── 📄 tasks.model.ts    # Main tasks model
│   │   │   └── 📄 tasks.repository.ts # Repository implementation
│   │   ├── 📄 tasks.controller.ts   # HTTP request handlers
│   │   ├── 📄 tasks.service.ts      # Business logic
│   │   └── 📄 tasks.module.ts       # Module configuration
│   ├── 📂 common/                   # Shared components
│   │   ├── 📂 filters/              # Exception filters
│   │   │   └── 📄 all-exceptions.filter.ts
│   │   └── 📂 pipes/                # Custom pipes
│   │       └── 📄 validation.pipe.ts
│   ├── 📄 app.module.ts             # Root module
│   └── 📄 main.ts                   # Application entry point
├── 📂 data/                         # Data storage
│   └── 📄 tasks.json                # JSON file database
├── 📂 test/                         # Test files
│   ├── 📂 tasks/                    # Feature tests
│   │   ├── 📄 tasks.controller.spec.ts
│   │   └── 📄 tasks.service.spec.ts
│   └── 📄 app.e2e-spec.ts           # End-to-end tests
├── 📄 package.json                  # Dependencies & scripts
├── 📄 tsconfig.json                 # TypeScript configuration
├── 📄 nest-cli.json                 # Nest.js CLI configuration
├── 📄 .env                          # Environment variables
├── 📄 .gitignore                    # Git ignore rules
└── 📄 README.md                     # Project documentation
```

## 🔧 Core Architectural Components

### 1. **Modules (Nest.js Foundation)**

**Purpose**: Organize related components into cohesive blocks
**Pattern**: Feature-based modular architecture

```typescript
// Example: Tasks Module
@Module({
  controllers: [TasksController],
  providers: [TasksService],
  exports: [TasksService],
})
export class TasksModule {}
```

**Benefits**:
- Encapsulation of related functionality
- Clear boundaries between features
- Easy testing and maintenance
- Reusable components

### 2. **Controllers (Gateway Layer)**

**Purpose**: Handle HTTP requests and responses
**Pattern**: RESTful API design

```typescript
@Controller('tasks')
export class TasksController {
  constructor(private readonly tasksService: TasksService) {}

  @Get()
  findAll(@Query() query: QueryTasksDto): Promise<Task[]> {
    return this.tasksService.findAll(query);
  }

  @Post()
  create(@Body() createTaskDto: CreateTaskDto): Promise<Task> {
    return this.tasksService.create(createTaskDto);
  }
}
```

**Responsibilities**:
- HTTP request routing
- Input validation (basic)
- Response formatting
- HTTP status code management
- Error propagation

### 3. **Services (Business Logic Layer)**

**Purpose**: Implement business logic and data operations
**Pattern**: Dependency injection, single responsibility

```typescript
@Injectable()
export class TasksService {
  constructor(private readonly fileRepository: FileRepository) {}

  async create(createTaskDto: CreateTaskDto): Promise<Task> {
    // Business logic implementation
    const newTask = this.buildTaskFromDto(createTaskDto);
    return await this.fileRepository.save(newTask);
  }
}
```

**Responsibilities**:
- Business rule enforcement
- Data transformation
- Orchestration of operations
- Transaction management

### 4. **Data Transfer Objects (DTOs)**

**Purpose**: Define data contracts for API operations
**Pattern**: Decorator-based validation

```typescript
export class CreateTaskDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(100)
  title: string;

  @IsString()
  @IsOptional()
  @MaxLength(500)
  description?: string;
}
```

**Benefits**:
- Type safety for inputs
- Automatic validation
- Clear API contracts
- Separation of concerns

### 5. **Entities (Data Models)**

**Purpose**: Define data structure and relationships
**Pattern**: Rich domain models

```typescript
export class Task {
  id: string;
  title: string;
  description?: string;
  status: TaskStatus;
  created_at: Date;
  updated_at: Date;
}
```

**Characteristics**:
- Strong typing
- Business invariants
- Data relationships
- Serialization logic

## 🔄 Data Flow Architecture

### Request Processing Flow

```
1. HTTP Request
   ↓
2. Route Matching (Controller)
   ↓
3. Input Validation (DTOs)
   ↓
4. Business Logic (Service)
   ↓
5. Data Persistence (Repository)
   ↓
6. Response Transformation
   ↓
7. HTTP Response
```

### Example: Create Task Request

```
Client Request:
POST /tasks
{
  "title": "Learn Nest.js",
  "description": "Complete the tutorial"
}

↓

Controller Validation:
✓ Valid JSON format
✓ Required fields present

↓

DTO Validation:
✓ Title: string, not empty, ≤100 chars
✓ Description: optional, ≤500 chars

↓

Service Logic:
✓ Generate UUID
✓ Set default status to 'pending'
✓ Set timestamps
✓ Business rule validation

↓

Data Persistence:
✓ Write to JSON file
✓ Atomic operation
✓ Error handling

↓

Response:
201 Created
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Learn Nest.js",
  "description": "Complete the tutorial",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00.000Z",
  "updated_at": "2024-01-15T10:30:00.000Z"
}
```

## 💾 Data Architecture

### Storage Strategy: JSON File System

**Choice Rationale**:
- ✅ **Simplicity**: No database setup required
- ✅ **Transparency**: Human-readable data format
- ✅ **Portability**: Easy backup and migration
- ✅ **Version Control**: Can track changes
- ✅ **Learning**: Focus on application logic

**Storage Structure**:
```json
{
  "tasks": [
    {
      "id": "uuid-string",
      "title": "Task title",
      "description": "Task description",
      "status": "pending|completed",
      "created_at": "ISO-8601-timestamp",
      "updated_at": "ISO-8601-timestamp"
    }
  ],
  "metadata": {
    "total_tasks": 1,
    "last_updated": "ISO-8601-timestamp",
    "version": "1.0.0"
  }
}
```

**Repository Pattern Implementation**:
```typescript
@Injectable()
export class FileRepository {
  private readonly filePath = './data/tasks.json';
  private tasks: Task[] = [];

  async findAll(): Promise<Task[]> { /* Implementation */ }
  async findById(id: string): Promise<Task | null> { /* Implementation */ }
  async save(task: Task): Promise<Task> { /* Implementation */ }
  async update(id: string, updates: Partial<Task>): Promise<Task> { /* Implementation */ }
  async delete(id: string): Promise<void> { /* Implementation */ }
}
```

### Data Integrity Strategies

1. **Atomic Operations**: Read-modify-write patterns
2. **Validation at Multiple Levels**: DTO, Service, Entity
3. **Backup Mechanisms**: File copies before major operations
4. **Error Recovery**: Rollback strategies for failed operations
5. **Concurrency Handling**: File locking for simultaneous access

## 🛡️ Security Architecture

### Input Validation Strategy

```
Multiple Validation Layers:

1. Controller Layer (Basic):
   ✓ HTTP method validation
   ✓ Content-Type checking
   ✓ Route parameter validation

2. DTO Layer (Comprehensive):
   ✓ Type checking with decorators
   ✓ Length restrictions
   ✓ Format validation
   ✓ Required field enforcement

3. Service Layer (Business):
   ✓ Business rule validation
   ✓ State consistency checks
   ✓ Permission validation (future)

4. Entity Layer (Data):
   ✓ Invariant maintenance
   ✓ Relationship validation
   ✓ Data format enforcement
```

### Error Handling Architecture

**Global Exception Filter**:
```typescript
@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();

    const status = this.getHttpStatus(exception);
    const message = this.getErrorMessage(exception);

    response.status(status).json({
      statusCode: status,
      timestamp: new Date().toISOString(),
      path: request.url,
      message,
    });
  }
}
```

**Error Response Format**:
```json
{
  "statusCode": 400,
  "timestamp": "2024-01-15T10:30:00.000Z",
  "path": "/tasks",
  "message": "Title is required and must be less than 100 characters"
}
```

## 🧪 Testing Architecture

### Test Pyramid Strategy

```
                    ┌─────────────────┐
                    │   E2E Tests     │  ← Few, High Value
                    │   (API Level)   │
                    └─────────────────┘
                ┌─────────────────────────┐
                │   Integration Tests    │  ← Moderate, Medium Value
                │   (Controller Level)   │
                └─────────────────────────┘
            ┌─────────────────────────────────┐
            │       Unit Tests                │  ← Many, Foundation
            │   (Service/Repository Level)    │
            └─────────────────────────────────┘
```

### Testing Patterns

1. **Unit Tests**:
   - Service logic isolation
   - Mock external dependencies
   - Fast execution, comprehensive coverage

2. **Integration Tests**:
   - Controller and service interaction
   - DTO validation
   - HTTP request/response cycles

3. **E2E Tests**:
   - Complete API workflows
   - File system integration
   - Real user scenarios

### Test Organization

```
test/
├── tasks/
│   ├── tasks.service.spec.ts      # Unit tests
│   ├── tasks.controller.spec.ts   # Integration tests
│   └── tasks.e2e-spec.ts         # End-to-end tests
├── common/
│   ├── filters.spec.ts           # Exception filter tests
│   └── pipes.spec.ts             # Validation pipe tests
└── app.e2e-spec.ts              # Application E2E tests
```

## 🚀 Performance Architecture

### Optimization Strategies

1. **Data Access Optimization**:
   - **In-memory caching**: Load JSON file once, cache in memory
   - **Lazy loading**: Read operations only when needed
   - **Batch operations**: Multiple operations in single file access

2. **Response Optimization**:
   - **Selective field return**: Only requested data
   - **Pagination support**: Large dataset handling
   - **Compression**: Response payload compression

3. **Request Processing**:
   - **Asynchronous operations**: Non-blocking I/O
   - **Connection pooling**: Efficient resource usage
   - **Request validation**: Early rejection of invalid requests

### Scalability Considerations

```
Current Architecture Limitations:
❌ Single JSON file → Concurrency bottleneck
❌ File system I/O → Performance constraints
❌ No database → Query optimization limitations

Future Scalability Path:
✅ Module-based design → Easy database migration
✅ Repository pattern → Swap storage implementation
✅ Dependency injection → Add caching layers
✅ Service abstraction → Microservices ready
```

## 🔄 Evolution Path

### Phase 1: Current (JSON File Storage)
- Simple CRUD operations
- Basic validation
- File-based persistence

### Phase 2: Enhanced Features
- Pagination and filtering
- Advanced search capabilities
- Bulk operations
- Data export/import

### Phase 3: Database Migration with Prisma + SQLite

#### **Technology Stack Upgrade**
```
JSON File Storage → Prisma ORM + SQLite Database
Nest.js v11.1.9 + Prisma v5.22.0
TypeScript 5.x with full type safety
```

#### **Prisma Schema Design**
```prisma
// prisma/schema.prisma
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

#### **Repository Pattern Evolution**
```typescript
// Before: JSON File Repository
@Injectable()
export class TasksRepository {
  private async saveTasks(): Promise<void> {
    await fs.writeFile(this.filePath, JSON.stringify(taskData));
  }
}

// After: Prisma Repository
@Injectable()
export class TasksRepository {
  constructor(private readonly prisma: PrismaService) {}

  async findAll(): Promise<Task[]> {
    return await this.prisma.task.findMany({
      orderBy: { createdAt: 'desc' }
    });
  }

  async findById(id: string): Promise<Task | null> {
    return await this.prisma.task.findUnique({
      where: { id }
    });
  }

  async create(data: CreateTaskDto): Promise<Task> {
    return await this.prisma.task.create({
      data: {
        ...data,
        status: data.status || TaskStatus.PENDING
      }
    });
  }

  async update(id: string, data: UpdateTaskDto): Promise<Task> {
    return await this.prisma.task.update({
      where: { id },
      data
    });
  }

  async delete(id: string): Promise<void> {
    await this.prisma.task.delete({
      where: { id }
    });
  }

  async search(query: string): Promise<Task[]> {
    return await this.prisma.task.findMany({
      where: {
        OR: [
          { title: { contains: query, mode: 'insensitive' } },
          { description: { contains: query, mode: 'insensitive' } }
        ]
      }
    });
  }

  async findByStatus(status: TaskStatus): Promise<Task[]> {
    return await this.prisma.task.findMany({
      where: { status },
      orderBy: { createdAt: 'desc' }
    });
  }
}
```

#### **Prisma Service Configuration**
```typescript
// src/prisma/prisma.service.ts
import { Injectable, OnModuleInit } from '@nestjs/common';
import { PrismaClient } from '@prisma/client';

@Injectable()
export class PrismaService extends PrismaClient implements OnModuleInit {
  async onModuleInit() {
    await this.$connect();
  }

  async onModuleDestroy() {
    await this.$disconnect();
  }

  async cleanDb() {
    await this.task.deleteMany();
  }
}
```

#### **Module Configuration Updates**
```typescript
// src/tasks/tasks.module.ts
import { Module } from '@nestjs/common';
import { TasksController } from './tasks.controller';
import { TasksService } from './tasks.service';
import { TasksRepository } from './models/tasks.repository';
import { PrismaService } from '../prisma/prisma.service';

@Module({
  controllers: [TasksController],
  providers: [
    TasksService,
    TasksRepository,
    PrismaService, // Add Prisma service
  ],
  exports: [TasksService, PrismaService],
})
export class TasksModule {}
```

#### **Migration Strategy**
```bash
# 1. Install Prisma dependencies
npm install prisma@5.22.0 @prisma/client@5.22.0
npm install -D @types/prisma

# 2. Initialize Prisma
npx prisma init --datasource-provider sqlite

# 3. Create schema (use schema design above)
# 4. Generate Prisma Client
npx prisma generate

# 5. Create initial migration
npx prisma migrate dev --name init

# 6. Data migration script (JSON → SQLite)
npx prisma db seed
```

#### **Data Migration Script**
```typescript
// prisma/seed.ts
import { PrismaClient, TaskStatus } from '@prisma/client';
import * as fs from 'fs/promises';

const prisma = new PrismaClient();

async function migrateFromJson() {
  try {
    // Read existing JSON data
    const data = await fs.readFile('./data/tasks.json', 'utf-8');
    const taskData = JSON.parse(data);

    // Migrate to database
    for (const task of taskData.tasks) {
      await prisma.task.create({
        data: {
          id: task.id,
          title: task.title,
          description: task.description,
          status: task.status === 'completed' ? TaskStatus.COMPLETED : TaskStatus.PENDING,
          createdAt: new Date(task.created_at),
          updatedAt: new Date(task.updated_at),
        }
      });
    }

    console.log(`✅ Migrated ${taskData.tasks.length} tasks to database`);
  } catch (error) {
    console.error('❌ Migration failed:', error);
  } finally {
    await prisma.$disconnect();
  }
}

migrateFromJson();
```

#### **Performance Benefits**

| Aspect | JSON File | Prisma + SQLite |
|--------|-----------|-----------------|
| **Read Performance** | O(n) linear search | O(log n) indexed queries |
| **Write Performance** | Full file rewrite | Single record operations |
| **Query Flexibility** | Manual filtering | Rich query API |
| **Type Safety** | Manual validation | Auto-generated types |
| **Data Integrity** | Manual checks | Database constraints |
| **Concurrency** | File locking required | Built-in transaction support |
| **Scalability** | Limited to 10k records | Millions of records possible |

#### **Advanced Query Examples**
```typescript
// Complex filtering and sorting
async findTasksWithFilters(query: QueryTasksDto): Promise<Task[]> {
  return await this.prisma.task.findMany({
    where: {
      status: query.status ? { equals: query.status } : undefined,
      OR: query.search ? [
        { title: { contains: query.search, mode: 'insensitive' } },
        { description: { contains: query.search, mode: 'insensitive' } }
      ] : undefined,
    },
    orderBy: [
      { [query.sortBy || 'createdAt']: query.sortOrder || 'desc' }
    ],
    skip: ((query.page || 1) - 1) * (query.limit || 10),
    take: query.limit || 10,
  });
}

// Aggregation queries
async getTaskStats(): Promise<TaskStats> {
  const [total, pending, completed] = await Promise.all([
    this.prisma.task.count(),
    this.prisma.task.count({ where: { status: TaskStatus.PENDING } }),
    this.prisma.task.count({ where: { status: TaskStatus.COMPLETED } })
  ]);

  return { total, pending, completed };
}

// Transaction support
async bulkUpdateStatus(ids: string[], status: TaskStatus): Promise<void> {
  await this.prisma.$transaction(
    ids.map(id =>
      this.prisma.task.update({
        where: { id },
        data: { status }
      })
    )
  );
}
```

#### **Production Migration Path**
```typescript
// Environment-specific configuration
const databaseConfig = {
  development: {
    provider: 'sqlite',
    url: './dev.db'
  },
  staging: {
    provider: 'postgresql',
    url: process.env.DATABASE_URL
  },
  production: {
    provider: 'postgresql',
    url: process.env.DATABASE_URL
  }
};

// Easy switch between storage backends
const useDatabase = process.env.USE_DATABASE === 'true';
const repository = useDatabase
  ? new PrismaTasksRepository()
  : new FileTasksRepository();
```

#### **Benefits Summary**

1. **Performance**: 10-100x faster queries with indexing
2. **Type Safety**: Compile-time guarantees with generated types
3. **Query Power**: Rich filtering, sorting, pagination, aggregation
4. **Data Integrity**: Database constraints and transactions
5. **Scalability**: From development to millions of records
6. **Developer Experience**: Auto-completion and query validation
7. **Production Ready**: Easy migration to PostgreSQL/MySQL

### Phase 4: Advanced Architecture
- Caching layer (Redis)
- Message queue for async operations
- API versioning
- Authentication/Authorization

### Phase 5: Microservices
- Separate user service
- Task processing service
- Notification service
- API Gateway

## 📏 Quality Standards

### Code Quality Metrics

1. **TypeScript Strict Mode**: Enabled
2. **ESLint Configuration**: Nest.js recommended rules
3. **Test Coverage**: Minimum 80%
4. **Code Documentation**: JSDoc comments
5. **Naming Conventions**: Consistent and descriptive

### Development Standards

1. **Git Workflow**: Feature branches, PR reviews
2. **CI/CD Pipeline**: Automated testing and deployment
3. **Code Reviews**: Peer review for all changes
4. **Documentation**: Updated with every feature
5. **Performance Monitoring**: Response time tracking

## 🎓 Architectural Learning Outcomes

By studying this architecture, developers will learn:

1. **Enterprise Patterns**: Dependency injection, modular design
2. **TypeScript Mastery**: Strong typing, decorators, generics
3. **REST API Design**: Best practices, HTTP semantics
4. **Testing Strategies**: Unit, integration, E2E testing
5. **File System Operations**: JSON persistence, atomic operations
6. **Error Handling**: Comprehensive exception management
7. **Security Principles**: Input validation, sanitization
8. **Performance Optimization**: Caching, efficient data access

## 🔍 Decision Rationale Summary

| Decision | Reasoning | Alternatives Considered |
|----------|-----------|-------------------------|
| **Nest.js** | Enterprise patterns, TypeScript first | Express.js, Fastify |
| **JSON Storage** | Simplicity, no database setup | SQLite, PostgreSQL |
| **DTO Pattern** | Type safety, validation | Direct object mapping |
| **Repository Pattern** | Data access abstraction | Direct file operations |
| **Modular Architecture** | Maintainability, testability | Monolithic structure |
| **Jest Testing** | Built-in with Nest.js, comprehensive | Mocha, Jasmine |

This architecture provides a solid foundation that balances **simplicity for learning** with **enterprise-grade patterns** for future growth and scalability.