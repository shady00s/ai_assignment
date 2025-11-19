# =� Task Manager REST API - Routing & Design Decisions

## =� Overview

This document outlines the key routing decisions, error handling strategies, and validation approaches implemented in the Task Manager REST API.

## =� Route Structure & Decisions

### Base Route Configuration

```
API Base URL: http://localhost:3000/api
Resource Endpoint: /tasks
```

**Decision Rationale:**
- **Global Prefix**: `/api` prefix for clear API separation from potential frontend routes
- **Resource Naming**: Plural `tasks` follows REST conventions for collection endpoints
- **Resource URLs**: Clean, intuitive paths that follow HTTP semantics

### Endpoints Overview

| Method | Endpoint | Purpose | Response Format |
|--------|----------|---------|-----------------|
| `GET` | `/api/tasks` | Retrieve all tasks with filtering | `Task[]` |
| `GET` | `/api/tasks/stats` | Get task statistics | `{total, pending, completed}` |
| `GET` | `/api/tasks/:id` | Retrieve specific task | `Task` |
| `POST` | `/api/tasks` | Create new task | `Task` |
| `PUT` | `/api/tasks/:id` | Update existing task | `Task` |
| `DELETE` | `/api/tasks/:id` | Delete task | `204 No Content` |

### Route Design Philosophy

1. **RESTful Compliance**: Each endpoint follows REST principles
   - GET for retrieval (safe operations)
   - POST for creation (non-idempotent)
   - PUT for updates (idempotent)
   - DELETE for removal (idempotent)

2. **Collection vs Resource Pattern**:
   - `/tasks` operates on the collection
   - `/tasks/:id` operates on specific resources

3. **HTTP Status Codes**:
   - `200 OK` - Successful GET operations
   - `201 Created` - Successful POST operations
   - `204 No Content` - Successful DELETE operations
   - `400 Bad Request` - Validation errors
   - `404 Not Found` - Resource not found
   - `500 Internal Server Error` - Server errors

## =� Error Handling Strategy

### Multi-Layer Error Handling

#### 1. **Global Exception Filter** (`AllExceptionsFilter`)
```typescript
@Catch()
export class AllExceptionsFilter implements ExceptionFilter
```

**Features:**
- **Comprehensive Coverage**: Catches all exceptions globally
- **Consistent Error Format**: Standardized error response structure
- **Detailed Logging**: Logs unhandled exceptions for debugging
- **HTTP Context Awareness**: Provides request details in errors

**Error Response Format:**
```json
{
  "statusCode": 400,
  "timestamp": "2024-01-15T10:30:00.000Z",
  "path": "/api/tasks",
  "method": "POST",
  "message": "title should not be empty, title must be a string"
}
```

#### 2. **Service Layer Error Handling**
```typescript
// Example: TasksService
if (!existingTask) {
  throw new NotFoundException(`Task with ID "${id}" not found`);
}

if (Object.keys(updateTaskDto).length === 0) {
  throw new BadRequestException('No valid fields provided for update');
}
```

**Error Types Used:**
- `NotFoundException` - Resource not found (404)
- `BadRequestException` - Invalid input/business rules (400)
- `UnauthorizedException` - Authentication failures (401) - *Future implementation*

#### 3. **Repository Layer Error Handling**
```typescript
try {
  const data = await fs.readFile(this.filePath, 'utf-8');
  const taskData: TaskData = JSON.parse(data);
  // Process data...
} catch (error) {
  this.tasks = [];
  await this.saveTasks(); // Create initial file if doesn't exist
}
```

### Error Handling Benefits

1. **User Experience**: Clear, actionable error messages
2. **Debugging**: Detailed error information for developers
3. **Security**: No sensitive information leaked in error responses
4. **Consistency**: All errors follow the same response format
5. **Maintainability**: Centralized error handling logic

##  Validation Strategy

### Three-Layer Validation Approach

#### 1. **DTO Layer Validation** (Input Sanitization)
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

  @IsEnum(TaskStatus)
  @IsOptional()
  status?: TaskStatus;
}
```

**Validation Rules:**
- **Type Safety**: `@IsString()`, `@IsEnum()` ensures correct data types
- **Presence Validation**: `@IsNotEmpty()` for required fields
- **Length Constraints**: `@MaxLength()` prevents oversized data
- **Optional Fields**: `@IsOptional()` for nullable data
- **Custom Validation**: Can be extended with custom validators

#### 2. **Global Validation Pipe**
```typescript
app.useGlobalPipes(
  new ValidationPipe({
    whitelist: true,           // Remove non-whitelisted properties
    forbidNonWhitelisted: true, // Reject unknown properties
    transform: true,           // Transform to DTO instances
    transformOptions: {
      enableImplicitConversion: true, // Auto-type conversion
    },
  }),
);
```

**Configuration Benefits:**
- **Security**: `whitelist` prevents parameter pollution
- **Strict Validation**: `forbidNonWhitelisted` catches typos
- **Convenience**: `transform` creates typed objects
- **Flexibility**: Implicit conversion for query parameters

#### 3. **Business Logic Validation** (Service Layer)
```typescript
// Example: Update validation
if (Object.keys(updateTaskDto).length === 0) {
  throw new BadRequestException('No valid fields provided for update');
}

// Example: Existence validation
const existingTask = await this.tasksRepository.findById(id);
if (!existingTask) {
  throw new NotFoundException(`Task with ID "${id}" not found`);
}
```

### Query Parameter Validation

```typescript
export class QueryTasksDto {
  @IsOptional()
  @IsEnum(TaskStatus)
  status?: TaskStatus;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(1)
  page?: number;

  @IsOptional()
  @IsString()
  search?: string;
}
```

**Query Validation Features:**
- **Type Conversion**: `@Type(() => Number)` converts string query params
- **Range Validation**: `@Min()`, `@Max()` for pagination limits
- **Enum Validation**: `@IsEnum()` for predefined values
- **Optional Parameters**: `@IsOptional()` for flexible querying

## =' Advanced Features Implementation

### 1. **Pagination & Sorting**
```typescript
const page = query.page || 1;
const limit = query.limit || 10;
const startIndex = (page - 1) * limit;
return tasks.slice(startIndex, startIndex + limit);
```

### 2. **Search Functionality**
```typescript
if (query.search) {
  const searchQuery = query.search.toLowerCase();
  tasks = tasks.filter(task =>
    task.title.toLowerCase().includes(searchQuery) ||
    (task.description && task.description.toLowerCase().includes(searchQuery))
  );
}
```

### 3. **Statistics Endpoint**
```typescript
async getStats(): Promise<{ total: number; pending: number; completed: number }> {
  const tasks = await this.tasksRepository.findAll();
  return {
    total: tasks.length,
    pending: tasks.filter(task => task.status === TaskStatus.PENDING).length,
    completed: tasks.filter(task => task.status === TaskStatus.COMPLETED).length,
  };
}
```

## 🗃️ **Prisma + SQLite Implementation Strategy**

### **Routing Impact with Database Storage**

The addition of Prisma + SQLite provides two implementation approaches while maintaining identical routing:

#### **Dual Storage Backend Strategy**
```typescript
// src/tasks/tasks.service.ts
@Injectable()
export class TasksService {
  constructor(
    @Inject('USE_JSON_STORAGE') private readonly useJsonStorage: boolean,
    private readonly jsonRepository: JsonTasksRepository,
    private readonly prismaRepository: PrismaTasksRepository,
  ) {}

  async findAll(query: QueryTasksDto): Promise<Task[]> {
    return this.useJsonStorage
      ? await this.jsonRepository.findAll(query)  // JSON implementation
      : await this.prismaRepository.findAll(query); // Prisma implementation
  }
}
```

### **Enhanced Query Capabilities with Prisma**

#### **Advanced Filtering Strategy**
```typescript
// Prisma-powered query with database optimizations
async findAll(query: QueryTasksDto): Promise<Task[]> {
  return await this.prisma.task.findMany({
    where: {
      // ✅ Database-level filtering (vs manual array filtering)
      status: query.status ? { equals: query.status } : undefined,
      OR: query.search ? [
        { title: { contains: query.search, mode: 'insensitive' } },
        { description: { contains: query.search, mode: 'insensitive' } }
      ] : undefined,
    },
    orderBy: [
      // ✅ Database-level sorting (vs manual array sorting)
      { [query.sortBy || 'createdAt']: query.sortOrder || 'desc' }
    ],
    // ✅ Database-level pagination (vs manual array slicing)
    skip: ((query.page || 1) - 1) * (query.limit || 10),
    take: query.limit || 10,
  });
}
```

#### **Route Enhancement: Statistics Endpoint**
```typescript
// JSON Implementation (current)
@Get('stats')
async getStats() {
  const tasks = await this.tasksService.findAll();
  return {
    total: tasks.length,
    pending: tasks.filter(task => task.status === 'pending').length,
    completed: tasks.filter(task => task.status === 'completed').length,
  };
}

// Prisma Implementation (optimized)
@Get('stats')
async getStats() {
  // ✅ Parallel database queries (vs sequential array processing)
  const [total, pending, completed] = await Promise.all([
    this.prisma.task.count(),
    this.prisma.task.count({ where: { status: 'PENDING' } }),
    this.prisma.task.count({ where: { status: 'COMPLETED' } }),
  ]);

  return { total, pending, completed };
}
```

### **Error Handling Evolution**

#### **Database-Specific Error Handling**
```typescript
async findById(id: string): Promise<Task> {
  try {
    const task = await this.prisma.task.findUnique({
      where: { id },
    });

    if (!task) {
      throw new NotFoundException(`Task with ID "${id}" not found`);
    }

    return task;
  } catch (error) {
    // ✅ Handle database-specific errors
    if (error.code === 'P2025') {
      throw new NotFoundException(`Task with ID "${id}" not found`);
    }
    if (error.code === 'P2002') {
      throw new ConflictException('Task already exists');
    }
    throw error; // Re-throw for global exception filter
  }
}
```

### **Transaction Support for Complex Routes**
```typescript
// Enhanced routes with database transactions
@Post('bulk')
async bulkCreate(@Body() createTaskDtos: CreateTaskDto[]): Promise<Task[]> {
  return await this.prisma.$transaction(
    createTaskDtos.map(dto =>
      this.prisma.task.create({ data: dto })
    )
  );
}

@Put('bulk/status')
async bulkUpdateStatus(
  @Body() { ids, status }: { ids: string[], status: TaskStatus }
): Promise<{ count: number }> {
  const result = await this.prisma.task.updateMany({
    where: { id: { in: ids } },
    data: { status },
  });

  return { count: result.count };
}
```

### **Backward Compatibility Strategy**

The routing architecture maintains **100% backward compatibility**:

1. **Identical Endpoints**: All routes remain `/api/tasks/*`
2. **Same Request/Response Formats**: DTOs and entities unchanged
3. **Consistent Error Handling**: Same error structure and status codes
4. **Gradual Migration**: Switch via `USE_JSON_STORAGE` environment variable

This dual-backend strategy allows teams to **start with JSON storage** and **migrate to Prisma** when performance requirements increase, without any API breaking changes.

## 🎯 Design Principles Applied

### 1. **SOLID Principles**
- **Single Responsibility**: Each class has one purpose
- **Open/Closed**: Easy to extend with new validation rules
- **Liskov Substitution**: DTOs work seamlessly with validation
- **Interface Segregation**: Focused interfaces for each concern
- **Dependency Inversion**: Services depend on abstractions

### 2. **HTTP/REST Best Practices**
- **Proper HTTP Methods**: GET, POST, PUT, DELETE
- **Status Code Semantics**: Appropriate codes for each scenario
- **Resource Naming**: Clear, consistent endpoint URLs
- **Content Negotiation**: JSON responses with proper headers

### 3. **Security Considerations**
- **Input Validation**: All inputs validated and sanitized
- **Error Information**: No sensitive data in error responses
- **Parameter Pollution**: Prevented via whitelist validation
- **Type Safety**: TypeScript prevents many runtime errors

### 4. **Developer Experience**
- **IntelliSense Support**: TypeScript provides full autocomplete
- **Clear Error Messages**: Actionable feedback for API consumers
- **Consistent Responses**: Predictable response structure
- **Documentation**: Self-documenting code with clear patterns

## = Future Enhancements

### Planned Improvements
1. **Authentication**: JWT-based auth with role-based access
2. **Advanced Filtering**: Date ranges, multiple status filters
3. **Bulk Operations**: Batch create, update, delete operations
4. **Caching**: Redis layer for improved performance
5. **Rate Limiting**: Prevent abuse of API endpoints
6. **API Versioning**: Support for multiple API versions

### Extensibility Points
- **Custom Validators**: Easy to add domain-specific validation
- **Middleware**: Authentication, logging, request timing
- **Database Migration**: Repository pattern abstracts storage layer
- **Event System**: Built-in Nest.js event handling for side effects

## =� Performance Considerations

### Current Optimizations
1. **In-Memory Caching**: JSON file loaded once at startup
2. **Lazy Loading**: File operations only when necessary
3. **Efficient Querying**: Array filtering with early returns
4. **Minimal Dependencies**: Lightweight dependency footprint

### Scalability Path
1. **Database Migration**: PostgreSQL/MongoDB integration
2. **Connection Pooling**: Efficient database connections
3. **Caching Layer**: Redis for frequently accessed data
4. **Load Balancing**: Multiple instances with sticky sessions

This routing and validation strategy provides a solid foundation that balances **security**, **performance**, and **developer experience** while maintaining clean separation of concerns and following industry best practices.