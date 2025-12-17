# Assignment: REST API for Task Management - AI Plan Mode Approach

## Original Assignment Prompt

**The Assignment Task: REST API for Task Management**

You will build a **RESTful Task Management API** with the following requirements:

### Core Functionality

• Create, read, update, and delete tasks (CRUD operations)

• Each task has: ID, title, description, status (pending/in-progress/completed), priority (low/medium/high), created timestamp

• Filter tasks by status and priority

• Data persistence (file-based or SQLite)

### Technical Requirements

• RESTful API design with proper HTTP methods (GET, POST, PUT, DELETE)

• Input validation and error handling

• Proper HTTP status codes

• API documentation (endpoint descriptions)

• At least 5 meaningful test cases

### Folder Structure

Your repository should contain:

```
task-api-vibe-coding/
|---- README.md # Project overview and setup instructions
|---- session3_model3/ # AI Plan Mode implementation
| |---- [code files]
| `---- prompts-session3.md # My prompts and AI responses
`---- report-session3.md # a written report
```

---

## AI Plan Mode Implementation - Session 3

### Approach Overview
This implementation uses the **AI Plan Mode** approach, where the AI first creates a comprehensive plan before executing any code changes. This emphasizes structured thinking, detailed planning, and systematic implementation.

## Prompting Strategy

### Initial Planning Phase

**User Prompt 1: Project Setup and Architecture Planning**
```
I need to create a RESTful Task Management API using NestJS with the following requirements:
- CRUD operations for tasks
- Task fields: ID, title, description, status (pending/in-progress/completed), priority (low/medium/high), created timestamp
- Filter tasks by status and priority
- SQLite data persistence
- Input validation and error handling
- Proper HTTP status codes
- API documentation
- At least 5 meaningful test cases

Before you start coding, I want you to create a detailed plan using AI Plan Mode. Analyze the requirements, design the architecture, and break down the implementation into clear, manageable steps. Consider:
1. Project structure and architecture
2. Database schema design
3. API endpoint design
4. DTO and validation strategy
5. Testing strategy
6. Documentation approach

Present this as a comprehensive plan before we proceed with implementation.
```

**AI Response 1: Comprehensive Architecture Plan**
The AI analyzed the requirements and provided a detailed plan covering:

1. **Project Architecture**
   - Traditional NestJS layered architecture
   - Controller → Service → Repository pattern
   - Prisma ORM for database operations
   - DTO pattern for data transfer and validation

2. **Database Schema Design**
   - Task entity with all required fields
   - Proper enums for status and priority
   - Timestamps for auditing
   - SQLite configuration for development

3. **API Endpoint Design**
   - GET /tasks - List all tasks with optional filtering
   - GET /tasks/:id - Get specific task
   - POST /tasks - Create new task
   - PATCH /tasks/:id - Update existing task
   - DELETE /tasks/:id - Delete task

4. **Implementation Phases**
   - Phase 1: Project setup and dependencies
   - Phase 2: Database schema and Prisma setup
   - Phase 3: DTOs and validation rules
   - Phase 4: Service layer implementation
   - Phase 5: Controller layer and routing
   - Phase 6: Testing implementation
   - Phase 7: Documentation and cleanup

### Implementation Phase Prompts

**User Prompt 2: Phase 1 Implementation**
```
Great plan! Now let's start with Phase 1: Project setup and dependencies. Create a new NestJS project and configure all necessary dependencies including:
- NestJS CLI and core modules
- Prisma with SQLite
- Validation libraries
- Testing frameworks
- TypeScript configuration

Please execute this step by step and explain what each dependency is for.
```

**AI Response 2: Project Setup Execution**
The AI proceeded with systematic setup:

1. Created NestJS project structure
2. Installed core dependencies:
   - `@nestjs/core`, `@nestjs/common`, `@nestjs/platform-express`
   - `prisma`, `@prisma/client` for database
   - `class-validator`, `class-transformer` for DTO validation
   - Testing libraries: `@nestjs/testing`, `jest`, `supertest`

3. Configured TypeScript and build tools
4. Set up Prisma with SQLite provider
5. Configured environment variables for database connection

**User Prompt 3: Database Schema Implementation**
```
Perfect! Now proceed with Phase 2: Database schema and Prisma setup. Design the Task entity with:
- All required fields (id, title, description, status, priority, createdAt, updatedAt)
- Proper enums for status and priority
- Appropriate constraints and validations
- Generate the initial migration

Show me the schema design and explain your choices.
```

**AI Response 3: Schema Design and Implementation**
The AI created a comprehensive Prisma schema:

```prisma
model Task {
  id          String      @id @default(cuid())
  title       String
  description String?
  status      TaskStatus  @default(PENDING)
  priority    TaskPriority @default(MEDIUM)
  createdAt   DateTime    @default(now())
  updatedAt   DateTime    @updatedAt

  @@map("tasks")
}

enum TaskStatus {
  PENDING
  IN_PROGRESS
  COMPLETED
}

enum TaskPriority {
  LOW
  MEDIUM
  HIGH
}
```

The AI explained the design decisions:
- `cuid()` for globally unique IDs
- Optional description field for flexibility
- Default values for status and priority
- Automatic timestamp management
- Clear enum definitions for status and priority

**User Prompt 4: DTOs and Validation Design**
```
Excellent! Now let's implement Phase 3: DTOs and validation rules. Create comprehensive DTOs for:
- CreateTaskDto with proper validation rules
- UpdateTaskDto with partial updates
- TaskFilterDto for query filtering
- TaskResponseDto for API responses

Ensure robust validation and clear error messages. Include type safety and documentation.
```

**AI Response 4: DTO Implementation**
The AI implemented a complete DTO system:

```typescript
// create-task.dto.ts
export class CreateTaskDto {
  @IsString()
  @IsNotEmpty()
  @MinLength(1)
  @MaxLength(255)
  title: string;

  @IsString()
  @IsOptional()
  @MaxLength(1000)
  description?: string;

  @IsEnum(TaskPriority)
  @IsOptional()
  priority?: TaskPriority = TaskPriority.MEDIUM;
}

// update-task.dto.ts
export class UpdateTaskDto extends PartialType(CreateTaskDto) {
  @IsEnum(TaskStatus)
  @IsOptional()
  status?: TaskStatus;
}

// task-filter.dto.ts
export class TaskFilterDto {
  @IsEnum(TaskStatus)
  @IsOptional()
  status?: TaskStatus;

  @IsEnum(TaskPriority)
  @IsOptional()
  priority?: TaskPriority;
}

// task-response.dto.ts
export class TaskResponseDto {
  id: string;
  title: string;
  description: string | null;
  status: TaskStatus;
  priority: TaskPriority;
  createdAt: Date;
  updatedAt: Date;

  static fromEntity(task: Task): TaskResponseDto {
    return {
      id: task.id,
      title: task.title,
      description: task.description,
      status: task.status,
      priority: task.priority,
      createdAt: task.createdAt,
      updatedAt: task.updatedAt,
    };
  }
}
```

**User Prompt 5: Service Layer Implementation**
```
Great work! Now proceed with Phase 4: Service layer implementation. Create the TasksService with:
- All CRUD operations (create, findAll, findOne, update, remove)
- Proper error handling for not found cases
- Business logic for filtering
- Database interaction through Prisma
- Type safety and proper return types

Focus on clean, maintainable code with proper separation of concerns.
```

**AI Response 5: Service Layer Implementation**
The AI implemented a comprehensive service layer (as seen in the actual code):

- Clean dependency injection of PrismaService
- Comprehensive CRUD methods with proper error handling
- Filtering logic in findAll method
- Consistent use of DTOs for data transformation
- Proper TypeScript typing throughout

**User Prompt 6: Testing Implementation**
```
Perfect! Now let's implement Phase 5: Testing. Create comprehensive tests for:
- Unit tests for TasksService methods
- Integration tests for TasksController endpoints
- Test database setup and teardown
- Mock data and edge cases
- Coverage for all business logic

Ensure at least 5 meaningful test cases as required by the assignment.
```

**AI Response 6: Testing Strategy**
The AI outlined a comprehensive testing approach:

1. **Unit Tests** for TasksService:
   - Test create task with valid data
   - Test find all tasks with/without filters
   - Test find one task (existing and non-existing)
   - Test update task with partial data
   - Test remove task

2. **Integration Tests** for TasksController:
   - Test POST /tasks endpoint
   - Test GET /tasks with filtering
   - Test GET /tasks/:id
   - Test PATCH /tasks/:id
   - Test DELETE /tasks/:id

3. **Edge Cases**:
   - Invalid input validation
   - Database constraint violations
   - Concurrent operations

### Key Observations from AI Plan Mode

## Strengths of Plan Mode Approach

1. **Structured Thinking**: The AI provided comprehensive planning before implementation
2. **Clear Roadmap**: Each phase built logically on the previous one
3. **Reduced Rework**: Planning prevented major architectural changes later
4. **Comprehensive Coverage**: All requirements were addressed systematically
5. **Documentation Integration**: Planning included documentation considerations

## Prompting Effectiveness

**Most Effective Prompts:**
- Phased implementation requests
- Specific architectural questions
- Validation rule specifications
- Testing strategy requests

**Less Effective Prompts:**
- Vague "implement everything" requests
- Open-ended design questions without context

## AI Decision-Making Process

The AI consistently:
1. Analyzed requirements thoroughly
2. Considered multiple implementation options
3. Chose industry-standard patterns
4. Explained design decisions
5. Maintained consistency across phases

## Challenges and Solutions

**Challenge 1: Initial Scope Management**
- Solution: Breaking down into clear, manageable phases

**Challenge 2: Validation Rule Complexity**
- Solution: Iterative refinement of DTO rules based on requirements

**Challenge 3: Testing Coverage**
- Solution: Comprehensive test planning covering unit and integration levels

This AI Plan Mode approach demonstrates how structured planning and systematic execution can lead to high-quality, maintainable code that fully meets assignment requirements.