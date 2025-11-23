# OptoPomodoro Monorepo Setup Guide

## Overview
This guide explains how to set up the OptoPomodoro monorepo structure using PNPM workspaces for optimal development workflow.

## Root Configuration Files

### 1. Root package.json
```json
{
  "name": "optopomodoro",
  "version": "1.0.0",
  "description": "Zen-inspired productivity application for Optomatica teams",
  "private": true,
  "workspaces": [
    "packages/*"
  ],
  "scripts": {
    "dev": "turbo run dev",
    "build": "turbo run build",
    "test": "turbo run test",
    "test:e2e": "turbo run test:e2e",
    "lint": "turbo run lint",
    "lint:fix": "turbo run lint:fix",
    "type-check": "turbo run type-check",
    "clean": "turbo run clean && rm -rf node_modules",
    "db:generate": "pnpm --filter backend db:generate",
    "db:push": "pnpm --filter backend db:push",
    "db:migrate": "pnpm --filter backend db:migrate",
    "db:seed": "pnpm --filter backend db:seed",
    "db:studio": "pnpm --filter backend db:studio",
    "docker:dev": "docker-compose -f docker-compose.dev.yml up",
    "docker:build": "docker-compose -f docker-compose.prod.yml build",
    "prepare": "husky install"
  },
  "devDependencies": {
    "@typescript-eslint/eslint-plugin": "^8.4.0",
    "@typescript-eslint/parser": "^8.4.0",
    "eslint": "^9.9.1",
    "eslint-config-prettier": "^9.1.0",
    "eslint-plugin-react": "^7.35.0",
    "eslint-plugin-react-hooks": "^4.6.2",
    "husky": "^9.1.5",
    "lint-staged": "^15.2.9",
    "prettier": "^3.3.3",
    "turbo": "^2.1.1",
    "typescript": "^5.5.4"
  },
  "engines": {
    "node": ">=18.0.0",
    "pnpm": ">=8.0.0"
  },
  "packageManager": "pnpm@9.6.0",
  "husky": {
    "hooks": {
      "pre-commit": "lint-staged",
      "pre-push": "pnpm type-check && pnpm test"
    }
  },
  "lint-staged": {
    "*.{js,jsx,ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,yml,yaml}": [
      "prettier --write"
    ]
  }
}
```

### 2. PNPM Workspace Configuration (pnpm-workspace.yaml)
```yaml
packages:
  - 'packages/*'
  - 'packages/*/src/**'
```

### 3. Turbo Configuration (turbo.json)
```json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "!.next/cache/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "test": {
      "dependsOn": ["build"],
      "outputs": ["coverage/**"],
      "cache": false
    },
    "test:e2e": {
      "dependsOn": ["build"],
      "cache": false
    },
    "lint": {
      "outputs": []
    },
    "lint:fix": {
      "outputs": []
    },
    "type-check": {
      "dependsOn": ["^build"],
      "outputs": []
    },
    "clean": {
      "cache": false
    },
    "db:generate": {
      "outputs": ["node_modules/.prisma/**"]
    },
    "db:push": {
      "cache": false
    },
    "db:migrate": {
      "cache": false
    },
    "db:seed": {
      "cache": false
    }
  }
}
```

### 4. TypeScript Configuration (tsconfig.json)
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "removeComments": false,
    "incremental": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noPropertyAccessFromIndexSignature": true,
    "exactOptionalPropertyTypes": true,
    "baseUrl": ".",
    "paths": {
      "@shared/*": ["packages/shared/src/*"],
      "@frontend/*": ["packages/frontend/src/*"],
      "@backend/*": ["packages/backend/src/*"]
    }
  },
  "include": ["packages/*/src/**/*"],
  "exclude": [
    "node_modules",
    "dist",
    "coverage",
    ".next",
    "packages/*/dist",
    "packages/*/coverage"
  ]
}
```

### 5. ESLint Configuration (.eslintrc.js)
```javascript
module.exports = {
  root: true,
  env: {
    browser: true,
    es2022: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'prettier',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: 2022,
    sourceType: 'module',
    project: './tsconfig.json',
  },
  plugins: ['react', 'react-hooks', '@typescript-eslint'],
  rules: {
    'react/react-in-jsx-scope': 'off',
    'react/prop-types': 'off',
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'off',
    '@typescript-eslint/no-explicit-any': 'warn',
    'prefer-const': 'error',
    'no-var': 'error',
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
  overrides: [
    {
      files: ['**/*.test.ts', '**/*.test.tsx'],
      rules: {
        '@typescript-eslint/no-explicit-any': 'off',
      },
    },
  ],
};
```

### 6. Prettier Configuration (.prettierrc)
```json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false,
  "endOfLine": "lf",
  "arrowParens": "always",
  "bracketSpacing": true,
  "bracketSameLine": false,
  "quoteProps": "as-needed",
  "jsxSingleQuote": true,
  "overrides": [
    {
      "files": "*.json",
      "options": {
        "singleQuote": false
      }
    },
    {
      "files": "*.md",
      "options": {
        "printWidth": 80,
        "proseWrap": "always"
      }
    }
  ]
}
```

### 7. Git Ignore (.gitignore)
```gitignore
# Dependencies
node_modules/
.pnpm-store/
.pnpm-debug.log*

# Build outputs
dist/
build/
.next/
out/

# Environment files
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Database
*.db
*.sqlite
*.sqlite3
packages/backend/data/

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*

# Runtime data
pids/
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/
*.lcov

# nyc test coverage
.nyc_output

# Grunt intermediate storage (https://gruntjs.com/creating-plugins#storing-task-files)
.grunt

# Bower dependency directory (https://bower.io/)
bower_components

# node-waf configuration
.lock-wscript

# Compiled binary addons (https://nodejs.org/api/addons.html)
build/Release

# Dependency directories
jspm_packages/

# TypeScript v1 declaration files
typings/

# TypeScript cache
*.tsbuildinfo

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache

# Optional stylelint cache
.stylelintcache

# Microbundle cache
.rpt2_cache/
.rts2_cache_cjs/
.rts2_cache_es/
.rts2_cache_umd/

# Optional REPL history
.node_repl_history

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity

# parcel-bundler cache (https://parceljs.org/)
.cache
.parcel-cache

# Next.js build output
.next

# Nuxt.js build / generate output
.nuxt
dist

# Storybook build outputs
.out
.storybook-out

# Temporary folders
tmp/
temp/

# Editor directories and files
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Turbo
.turbo

# Docker
.dockerignore

# Prisma
packages/backend/prisma/migrations/
packages/backend/node_modules/.prisma/

# Testing
coverage/
.nyc_output/
```

### 8. Docker Development Environment (docker-compose.dev.yml)
```yaml
version: '3.8'

services:
  frontend:
    build:
      context: .
      dockerfile: docker/frontend.Dockerfile
      target: development
    ports:
      - "3000:3000"
    volumes:
      - ./packages/frontend:/app/packages/frontend
      - /app/packages/frontend/node_modules
    environment:
      - VITE_API_URL=http://localhost:3001
      - VITE_WS_URL=ws://localhost:3001
      - NODE_ENV=development
    depends_on:
      - backend
    command: pnpm dev

  backend:
    build:
      context: .
      dockerfile: docker/backend.Dockerfile
      target: development
    ports:
      - "3001:3001"
    volumes:
      - ./packages/backend:/app/packages/backend
      - /app/packages/backend/node_modules
      - backend_data:/app/packages/backend/data
    environment:
      - NODE_ENV=development
      - DATABASE_URL=file:./data/optopomodoro.db
      - JWT_SECRET=dev-super-secret-jwt-key-change-in-production
      - PORT=3001
      - FRONTEND_URL=http://localhost:3000
    command: pnpm dev

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  backend_data:
  redis_data:
```

### 9. Environment Variables Template (.env.example)
```bash
# Application
NODE_ENV=development
PORT=3001
FRONTEND_URL=http://localhost:3000
PRODUCTION_URL=https://optopomodoro.com

# Database
DATABASE_URL=file:./data/optopomodoro.db

# JWT
JWT_SECRET=your-super-secret-jwt-key-change-in-production
JWT_EXPIRES_IN=7d

# OAuth (for future implementation)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_CALLBACK_URL=http://localhost:3001/api/auth/google/callback

MICROSOFT_CLIENT_ID=your-microsoft-client-id
MICROSOFT_CLIENT_SECRET=your-microsoft-client-secret
MICROSOFT_CALLBACK_URL=http://localhost:3001/api/auth/microsoft/callback

# Redis (for caching and sessions)
REDIS_URL=redis://localhost:6379

# External Services (for future integrations)
SLACK_BOT_TOKEN=your-slack-bot-token
CALENDAR_API_KEY=your-calendar-api-key

# Email (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@optomatica.com
SMTP_PASS=your-app-password
```

### 10. README.md
```markdown
# OptoPomodoro

Zen-inspired productivity application for Optomatica teams built with modern web technologies.

## 🚀 Tech Stack

- **Frontend**: React 18 + TypeScript + Vite + PWA
- **Backend**: NestJS + TypeScript + SQLite + Prisma
- **Real-time**: Socket.IO
- **Styling**: Styled-components with Design System
- **State Management**: Redux Toolkit + RTK Query
- **Testing**: Vitest + React Testing Library + Cypress
- **Build System**: Turbo + PNPM Workspaces

## 📁 Project Structure

```
optopomodoro/
├── packages/
│   ├── frontend/          # React PWA application
│   ├── backend/           # NestJS API server
│   └── shared/            # Shared types and utilities
├── docker/                # Docker configurations
├── docs/                  # Project documentation
├── scripts/               # Build and deployment scripts
├── package.json           # Root package configuration
├── pnpm-workspace.yaml    # PNPM workspace configuration
├── turbo.json             # Turbo build system configuration
└── README.md              # This file
```

## 🛠️ Development Setup

### Prerequisites

- Node.js 18+
- PNPM 8+
- Docker (optional, for containerized development)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd optopomodoro
```

2. Install dependencies:
```bash
pnpm install
```

3. Copy environment variables:
```bash
cp .env.example .env
```

4. Initialize database:
```bash
pnpm db:generate
pnpm db:push
pnpm db:seed
```

### Development

Start the development servers:

```bash
# Start both frontend and backend
pnpm dev

# Or start individually
pnpm --filter frontend dev
pnpm --filter backend dev
```

### Using Docker

```bash
# Start all services with Docker
pnpm docker:dev
```

## 📜 Available Scripts

- `pnpm dev` - Start development servers
- `pnpm build` - Build all packages
- `pnpm test` - Run all tests
- `pnpm test:e2e` - Run E2E tests
- `pnpm lint` - Lint all packages
- `pnpm type-check` - Type check all packages
- `pnpm clean` - Clean all build artifacts
- `pnpm db:studio` - Open Prisma Studio
- `pnpm docker:dev` - Start development with Docker

## 📊 Testing

- Unit Tests: Vitest + React Testing Library
- Integration Tests: Supertest (Backend)
- E2E Tests: Cypress
- Coverage Goal: 80%+

## 🚀 Deployment

The application is designed for easy deployment to various platforms:

- **Vercel/Netlify**: Frontend PWA
- **Railway/Render**: Backend API
- **Docker**: Full application stack

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is proprietary to Optomatica Inc.

## 🤝 Support

For support, please contact the development team at dev@optomatica.com.
```

## Package-Specific package.json Files

### Frontend package.json (packages/frontend/package.json)
```json
{
  "name": "@optopomodoro/frontend",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:e2e": "cypress run",
    "test:e2e:open": "cypress open",
    "lint": "eslint src --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "lint:fix": "eslint src --ext ts,tsx --fix",
    "type-check": "tsc --noEmit",
    "clean": "rm -rf dist node_modules"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.1",
    "@reduxjs/toolkit": "^2.2.5",
    "react-redux": "^9.1.2",
    "@reduxjs/react-persist": "^2.1.0",
    "styled-components": "^6.1.11",
    "framer-motion": "^11.3.19",
    "idb": "^8.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@types/styled-components": "^5.1.34",
    "@vitejs/plugin-react": "^4.3.1",
    "vite": "^5.4.2",
    "vite-plugin-pwa": "^0.20.1",
    "workbox-window": "^7.1.0",
    "vitest": "^2.0.5",
    "@vitest/ui": "^2.0.5",
    "@testing-library/react": "^16.0.0",
    "@testing-library/jest-dom": "^6.5.0",
    "@testing-library/user-event": "^14.5.2",
    "cypress": "^13.13.3",
    "@cypress/react": "^6.0.0"
  }
}
```

### Backend package.json (packages/backend/package.json)
```json
{
  "name": "@optopomodoro/backend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "nest start --watch",
    "build": "nest build",
    "start": "node dist/main",
    "test": "jest",
    "test:e2e": "jest --config ./test/jest-e2e.json",
    "test:watch": "jest --watch",
    "test:cov": "jest --coverage",
    "test:debug": "node --inspect-brk -r tsconfig-paths/register -r ts-node/register node_modules/.bin/jest --runInBand",
    "test:e2e:debug": "node --inspect-brk -r tsconfig-paths/register -r ts-node/register node_modules/.bin/jest --config ./test/jest-e2e.json --runInBand",
    "lint": "eslint \"{src,apps,libs,test}/**/*.ts\" --fix",
    "type-check": "tsc --noEmit",
    "clean": "rm -rf dist node_modules",
    "db:generate": "prisma generate",
    "db:push": "prisma db push",
    "db:migrate": "prisma migrate dev",
    "db:seed": "ts-node prisma/seed.ts",
    "db:studio": "prisma studio"
  },
  "dependencies": {
    "@nestjs/core": "^10.4.1",
    "@nestjs/common": "^10.4.1",
    "@nestjs/platform-express": "^10.4.1",
    "@nestjs/config": "^3.2.3",
    "@nestjs/jwt": "^10.2.0",
    "@nestjs/passport": "^10.0.3",
    "@nestjs/platform-socket.io": "^10.4.1",
    "@nestjs/throttler": "^5.1.2",
    "prisma": "^5.19.1",
    "@prisma/client": "^5.19.1",
    "passport": "^0.7.0",
    "@passport-jwt": "^0.4.1",
    "passport-google-oauth20": "^2.0.0",
    "bcryptjs": "^2.4.3",
    "helmet": "^7.1.0",
    "class-validator": "^0.14.1",
    "class-transformer": "^0.5.1",
    "socket.io": "^4.7.5",
    "winston": "^3.13.1",
    "@nestjs/winston": "^2.0.4",
    "date-fns": "^3.6.0",
    "uuid": "^10.0.0"
  },
  "devDependencies": {
    "@nestjs/cli": "^10.4.4",
    "@nestjs/schematics": "^10.1.1",
    "@nestjs/testing": "^10.4.1",
    "@types/passport-jwt": "^4.0.1",
    "@types/passport-google-oauth20": "^2.0.16",
    "@types/bcryptjs": "^2.4.6",
    "@types/uuid": "^10.0.0",
    "jest": "^29.7.0",
    "@types/jest": "^29.5.12",
    "supertest": "^6.3.3",
    "@types/supertest": "^6.0.2",
    "ts-node": "^10.9.2",
    "tsconfig-paths": "^4.2.0"
  }
}
```

This monorepo setup provides:

1. **Optimized development workflow** with PNPM workspaces and Turbo
2. **Consistent tooling** across all packages (ESLint, Prettier, TypeScript)
3. **Efficient caching** and parallel builds with Turbo
4. **Docker support** for containerized development
5. **Comprehensive testing setup** with unit, integration, and E2E tests
6. **Pre-commit hooks** for code quality enforcement
7. **Environment management** with templates and examples
8. **Scalable structure** ready for future growth

The configuration enables both teams to work efficiently while maintaining code quality, consistency, and optimal build performance.