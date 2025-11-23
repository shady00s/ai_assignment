# OptoPomodoro Frontend

A Zen-inspired productivity application built for Optomatica employees. This is the frontend package of the OptoPomodoro monorepo.

## Overview

OptoPomodoro is a modern task management application that combines the Pomodoro Technique with mindfulness principles. Built with React 18, TypeScript, and a mobile-first approach, it provides a serene yet powerful productivity workspace.

## Key Features

### 🍅 **Pomodoro Timer**
- Customizable work/break durations
- Mindfulness integration with wellness reminders
- Ambient sound options (forest, ocean, cafe, rain)
- Progress tracking with quality ratings

### 📋 **Task Management**
- Visual Kanban board with drag-and-drop
- Task prioritization (LOW, MEDIUM, HIGH, URGENT)
- Pomodoro estimation and tracking
- Mobile-optimized touch gestures

### 🎮 **Gamification**
- XP system and level progression
- Achievement unlocking
- Productivity streaks
- Wellness score tracking

### 👥 **Team Features**
- Real-time collaboration via WebSocket
- Team challenges and leaderboards
- Shared task boards
- Member presence indicators

### 📱 **Mobile Excellence**
- Progressive Web App (PWA)
- Touch-optimized interface
- Swipe navigation for Kanban columns
- Offline capability

## Technology Stack

- **React 18** - Modern hooks and concurrent features
- **TypeScript** - Type safety and enhanced DX
- **Vite** - Lightning-fast development and builds
- **Redux Toolkit** - State management with RTK Query
- **Styled Components** - CSS-in-JS with theme system
- **React DnD** - Drag and drop functionality
- **Framer Motion** - Smooth animations
- **Socket.IO Client** - Real-time communication
- **PWA** - Native-like mobile experience

## Architecture

### Component Structure
```
src/
├── components/
│   ├── atoms/           # Reusable UI elements
│   ├── molecules/       # Composed components
│   ├── organisms/       # Complex components
│   ├── templates/       # Page layouts
│   └── pages/           # Complete page views
├── hooks/              # Custom React hooks
├── store/              # Redux state management
│   ├── slices/         # Feature-specific state
│   └── api/            # RTK Query endpoints
├── themes/             # Design tokens and themes
├── types/              # TypeScript definitions
└── utils/              # Utility functions
```

### State Management
- **Redux Toolkit** for global state
- **RTK Query** for server state and caching
- **React Query patterns** for optimistic updates
- **WebSocket integration** for real-time sync

### Styling System
- **Design tokens** for consistency
- **Mobile-first responsive design**
- **Zen-inspired color palette**
- **Accessibility-first approach**

## Development

### Prerequisites
- Node.js 18+
- pnpm (preferred package manager)

### Setup
```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Run tests
pnpm test

# Build for production
pnpm build

# Preview production build
pnpm preview
```

### Environment Variables
```bash
VITE_API_URL=http://localhost:3001
VITE_WS_URL=ws://localhost:3001
VITE_APP_NAME=OptoPomodoro
```

## Available Scripts

- `pnpm dev` - Start development server
- `pnpm build` - Build for production
- `pnpm preview` - Preview production build
- `pnpm test` - Run tests with Vitest
- `pnpm test:ui` - Run tests with Vitest UI
- `pnpm test:e2e` - Run E2E tests with Cypress
- `pnpm test:e2e:open` - Open Cypress E2E test runner
- `pnpm lint` - Run ESLint
- `pnpm lint:fix` - Fix linting issues
- `pnpm type-check` - Run TypeScript type checking

## PWA Features

This application is a Progressive Web App (PWA) that offers:

- **Offline Support** - Core functionality works offline
- **App-like Experience** - Installable on mobile devices
- **Push Notifications** - Timers and wellness reminders
- **Background Sync** - Data synchronization when online

## Mobile Optimization

- **Touch Gestures** - Swipe between Kanban columns
- **Responsive Design** - Optimized for all screen sizes
- **Performance** - 60fps animations and smooth interactions
- **Accessibility** - WCAG 2.1 AA compliant

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- iOS Safari 14+
- Android Chrome 90+

## Contributing

1. Follow the existing code style
2. Use TypeScript strictly
3. Write tests for new features
4. Ensure mobile responsiveness
5. Test accessibility

## License

© 2024 Optomatica. All rights reserved.

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
