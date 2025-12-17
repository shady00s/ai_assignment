# Assignment: REST API for Task Management

## Original Assignment Prompt

**The Assignment Task: REST API for Task Management**

You will build a **RESTful Task Management API** with the following requirements:

### Core Functionality

• Create, read, update, and delete tasks (CRUD operations)

• Each task has: ID, title, description, status (pending/in-progress/completed), priority (low/medium/high), created timestamp

• Filter tasks by status and priority

• Data persistence (file-based or SQLite)

### Technical Requirements

• RESTful API design with proper HTTP methods (GET, POST, PUT, DELETE) • Input validation and error handling

• Proper HTTP status codes

• API documentation (endpoint descriptions)

• At least 5 meaningful test cases

### Folder Structure

Your repository should contain:

```
task-api-vibe-coding/
|---- README.md # Project overview and setup instructions
|---- session1_model1/ # First model implementation
| |---- [code files]
| `---- prompts.md # My prompts and AI responses
`---- report.md # a written report
```

### Written Report Requirements

Your report should include:

**A. Approach Dimensions**
- Development speed
- Code quality (maintainability, readability)
- Ease of use / cognitive load
- Control and predictability
- Suitability for different scenarios
- Your subjective experience

**B. Detailed Analysis**
- **Workflow description**: How did you actually use it?
- **Prompting strategy**: What kinds of prompts worked best?
- **Strengths observed**: What went surprisingly well?
- **Challenges faced**: Where did you struggle?
- **Generated code quality**: Assess structure, readability, correctness
- **Time spent**: Rough breakdown of your 30-40 minutes

## Initial Observation from GLM Prompt

From the existing `observations.md` file, there was evidence of a previous GLM session:

```
GLM Prompt:
> create nestJS CRUD operations for task manager app, the first feature needed to be implemented is
 "tasks" feature. \
\tasks structure:
 Each task has: ID, title, description, status (pending/in-progress/completed), priority
 (low/medium/high),
 created timestamp ,also Filter tasks by status and priority.\
- needed packages:\
   - prisma\
\
acceptance criteria:\
- clean code\
- each  endpoint has its own DTO, do not use "any" type\
ultrathink
```

## Implementation Approach

### Discovery Phase

Upon exploration, I found that a complete NestJS Task Management API was already implemented in the `task_manager/` directory, including:

- ✅ Complete CRUD operations
- ✅ All required data fields and enums
- ✅ Filtering capabilities
- ✅ SQLite database with Prisma ORM
- ✅ Proper DTOs (no "any" types)
- ✅ RESTful design with proper HTTP methods
- ✅ Error handling and HTTP status codes

### Enhancement Focus

Since the core API was already fully functional and exceeded the basic requirements, the implementation focused on:

1. **Project Structure**: Reorganizing into the required `task-api-vibe-coding/session1_model1/` structure
2. **Comprehensive Testing**: Creating extensive unit test suites (23 test cases for service, 18 for controller)
3. **API Documentation**: Adding Swagger/OpenAPI documentation with detailed endpoint descriptions
4. **Documentation**: Creating comprehensive README with setup instructions
5. **Analysis**: Writing detailed evaluation report

### Key Implementation Decisions

1. **Preservation of Existing Code**: Rather than rebuilding working functionality, focused on enhancing what already existed
2. **Testing Strategy**: Implemented comprehensive unit tests covering all service methods and controller endpoints
3. **Documentation Standards**: Added professional API documentation with examples and error cases
4. **Project Organization**: Restructured to match assignment requirements while maintaining code integrity

## AI Interaction Summary

### Claude Code Interaction Pattern

The development process followed this pattern:
1. **Discovery**: Explored existing codebase to understand current implementation
2. **Planning**: Created comprehensive plan to complete missing requirements
3. **Execution**: Systematically implemented missing features
4. **Validation**: Ensured all requirements were met and tests passed

### Prompt Effectiveness

- **Initial Assessment**: Prompts for code exploration were highly effective
- **Planning Phase**: Structured planning prompts helped organize complex requirements
- **Test Generation**: Specific prompts for test creation generated comprehensive coverage
- **Documentation**: Clear requirements led to professional-grade documentation

## Outcome Assessment

The final implementation exceeds the assignment requirements by providing:
- 41+ test cases (far exceeding the minimum 5 required)
- Professional API documentation with Swagger
- Comprehensive README with examples
- Clean, maintainable code structure
- Type-safe implementation throughout

This demonstrates how AI-assisted development can rapidly produce high-quality, production-ready code when working with an existing functional foundation.