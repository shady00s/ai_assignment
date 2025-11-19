# Task Manager REST API

A robust REST API for managing tasks with dual storage backend support (JSON file and SQLite database with Prisma ORM).

## Features

- ✅ **Dual Storage Backend**: Switch between JSON file storage and SQLite database
- ✅ **CRUD Operations**: Create, Read, Update, Delete tasks
- ✅ **Advanced Querying**: Search, filter, sort, and paginate tasks
- ✅ **Bulk Operations**: Create and update multiple tasks simultaneously
- ✅ **Data Validation**: Input validation with comprehensive error handling
- ✅ **Statistics**: Get task statistics (total, pending, completed)
- ✅ **Data Migration**: Seamless migration from JSON to SQLite database
- ✅ **Type Safety**: Full TypeScript support with auto-generated database types

## Architecture

The API follows a layered architecture with clean separation of concerns:

```
src/
├── tasks/
│   ├── controllers/     # HTTP request handling
│   ├── services/        # Business logic
│   ├── repositories/    # Data access layer
│   ├── entities/        # Data models
│   └── dto/            # Data transfer objects
├── prisma/              # Database configuration and migration
└── main.ts             # Application entry point
```

## Storage Backend Options

### 1. JSON File Storage
- Simple file-based storage
- No database setup required
- Perfect for development and small datasets
- Data persistence in `data/tasks.json`

### 2. SQLite Database with Prisma ORM
- Robust database storage
- Advanced querying capabilities
- Bulk operations support
- Better performance for large datasets
- Data migration from JSON supported

## Quick Start

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd task-manager-api
```

2. Install dependencies
```bash
npm install
```

3. Configure storage backend (see Configuration section)

4. For database storage (optional):
```bash
# Generate Prisma client
npm run db:generate

# Run database migrations
npm run db:migrate

# Migrate existing JSON data (if any)
npm run db:seed
```

5. Start the development server
```bash
npm run start:dev
```

The API will be available at `http://localhost:3000`

## Configuration

### Environment Variables (.env)

```env
# Server Configuration
PORT=3000
NODE_ENV=development

# Storage Configuration (Choose one)
USE_JSON_STORAGE=false
DATABASE_URL="file:./dev.db"

# File Storage (for JSON storage)
DATA_FILE_PATH=./data/tasks.json

# Validation
MAX_TITLE_LENGTH=100
MAX_DESCRIPTION_LENGTH=500
```

### Switching Storage Backends

**JSON Storage:**
```env
USE_JSON_STORAGE=true
```

**Database Storage:**
```env
USE_JSON_STORAGE=false
DATABASE_URL="file:./dev.db"
```

## API Endpoints

### Tasks

#### Get All Tasks
```http
GET /tasks
```

**Query Parameters:**
- `page` - Page number (default: 1)
- `limit` - Items per page (default: 10)
- `sortBy` - Sort field (default: createdAt)
- `sortOrder` - Sort order: 'asc' or 'desc' (default: desc)
- `status` - Filter by status: PENDING or COMPLETED
- `search` - Search in title and description

**Example:**
```http
GET /tasks?page=1&limit=5&status=PENDING&search=important
```

#### Get Task by ID
```http
GET /tasks/:id
```

#### Create Task
```http
POST /tasks
```

**Request Body:**
```json
{
  "title": "Complete project documentation",
  "description": "Write comprehensive README and API documentation",
  "status": "PENDING"
}
```

#### Update Task
```http
PUT /tasks/:id
```

**Request Body:**
```json
{
  "title": "Updated task title",
  "description": "Updated description",
  "status": "COMPLETED"
}
```

#### Delete Task
```http
DELETE /tasks/:id
```

### Bulk Operations

#### Bulk Create Tasks
```http
POST /tasks/bulk
```

**Request Body:**
```json
[
  {
    "title": "Task 1",
    "description": "Description 1"
  },
  {
    "title": "Task 2",
    "description": "Description 2",
    "status": "PENDING"
  }
]
```

#### Bulk Update Status
```http
PUT /tasks/bulk/status
```

**Request Body:**
```json
{
  "ids": ["task-id-1", "task-id-2"],
  "status": "COMPLETED"
}
```

### Statistics

#### Get Task Statistics
```http
GET /tasks/stats
```

**Response:**
```json
{
  "total": 150,
  "pending": 75,
  "completed": 75
}
```

### Storage Backend Info

#### Get Current Storage Backend
```http
GET /tasks/storage-backend
```

**Response:**
```json
{
  "backend": "database",
  "message": "Currently using database storage backend"
}
```

## Data Models

### Task Entity
```typescript
interface Task {
  id: string;
  title: string;
  description: string | null;
  status: 'PENDING' | 'COMPLETED';
  created_at: Date;
  updated_at: Date;
}
```

### Task Status
- `PENDING` - Task is yet to be completed
- `COMPLETED` - Task has been finished

## Database Schema (SQLite)

The database uses the following schema:

```sql
CREATE TABLE tasks (
  id          TEXT     PRIMARY KEY,
  title       TEXT     NOT NULL,
  description TEXT,
  status      TEXT     DEFAULT 'PENDING',
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX tasks_status_idx ON tasks(status);
CREATE INDEX tasks_created_at_idx ON tasks(created_at);
CREATE INDEX tasks_title_idx ON tasks(title);
```

## Database Management

### Available Commands

```bash
# Generate Prisma client
npm run db:generate

# Create and run migrations
npm run db:migrate

# Open Prisma Studio (database GUI)
npm run db:studio

# Migrate data from JSON to SQLite
npm run db:seed

# Reset database (dangerous - deletes all data)
npm run db:reset
```

### Data Migration

The project includes a built-in migration tool to transfer data from JSON storage to SQLite database:

```bash
npm run db:seed
```

This script will:
1. Read existing data from `data/tasks.json`
2. Clear the database (optional)
3. Migrate all tasks with proper status mapping
4. Preserve timestamps
5. Report migration statistics

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages:

- `400 Bad Request` - Validation errors
- `404 Not Found` - Task not found
- `409 Conflict` - Duplicate task or data integrity issue
- `500 Internal Server Error` - Server errors

### Error Response Format
```json
{
  "statusCode": 404,
  "message": "Task with ID \"123\" not found",
  "error": "Not Found"
}
```

## Development

### Running Tests
```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:cov

# Run end-to-end tests
npm run test:e2e
```

### Code Quality
```bash
# Lint code
npm run lint

# Format code
npm run format
```

### Building for Production
```bash
# Build the project
npm run build

# Start production server
npm run start:prod
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the ISC License.

## API Examples

### Creating a Task
```bash
curl -X POST http://localhost:3000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Learn NestJS",
    "description": "Complete NestJS tutorial",
    "status": "PENDING"
  }'
```

### Getting Tasks with Filters
```bash
curl "http://localhost:3000/tasks?status=PENDING&limit=5&sortBy=createdAt&sortOrder=desc"
```

### Bulk Creating Tasks
```bash
curl -X POST http://localhost:3000/tasks/bulk \
  -H "Content-Type: application/json" \
  -d '[
    {
      "title": "Setup project",
      "description": "Initialize project structure"
    },
    {
      "title": "Write tests",
      "description": "Create unit and integration tests"
    }
  ]'
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   - Ensure `DATABASE_URL` is correctly configured
   - Run `npm run db:migrate` to create database tables
   - Check file permissions for database file

2. **Prisma Client Generation Issues**
   - Run `npm run db:generate` to regenerate client
   - Ensure Prisma schema is valid

3. **Migration Issues**
   - Backup your data before running migrations
   - Check JSON file format before migrating
   - Run `npm run db:reset` to start fresh (deletes all data)

### Logs

Check console output for detailed logs:
- Database connection status
- Migration progress
- Storage backend information
- Error details