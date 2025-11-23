import { PrismaClient } from '@prisma/client';
import * as bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Starting database seed...');

  // Create admin user
  const adminPassword = await bcrypt.hash('admin123', 12);
  const adminUser = await prisma.user.upsert({
    where: { email: 'admin@optopomodoro.com' },
    update: {},
    create: {
      email: 'admin@optopomodoro.com',
      passwordHash: adminPassword,
      firstName: 'Admin',
      lastName: 'User',
      level: 10,
      xp: 5000,
      streak: 30,
      totalFocusTime: 3600, // 60 hours
      tasksCompleted: 100,
      qualityScore: 4.5,
      wellnessScore: 4.2,
      preferences: JSON.stringify({
        theme: 'dark',
        pomodoroLength: 25,
        shortBreakLength: 5,
        longBreakLength: 15,
        notifications: {
          desktop: true,
          sound: true,
          email: false,
        },
        autoStartBreaks: false,
        autoStartPomodoros: false,
      }),
    },
  });

  console.log('✅ Created admin user:', adminUser.email);

  // Create demo users
  const demoPassword = await bcrypt.hash('demo123', 12);
  const demoUser = await prisma.user.upsert({
    where: { email: 'demo@optopomodoro.com' },
    update: {},
    create: {
      email: 'demo@optopomodoro.com',
      passwordHash: demoPassword,
      firstName: 'Demo',
      lastName: 'User',
      level: 5,
      xp: 1500,
      streak: 7,
      totalFocusTime: 1200, // 20 hours
      tasksCompleted: 35,
      qualityScore: 4.0,
      wellnessScore: 3.8,
      preferences: JSON.stringify({
        theme: 'light',
        pomodoroLength: 25,
        shortBreakLength: 5,
        longBreakLength: 15,
        notifications: {
          desktop: true,
          sound: false,
          email: false,
        },
        autoStartBreaks: true,
        autoStartPomodoros: false,
      }),
    },
  });

  console.log('✅ Created demo user:', demoUser.email);

  // Create demo team
  const demoTeam = await prisma.team.upsert({
    where: { id: 'demo-team' },
    update: {},
    create: {
      id: 'demo-team',
      name: 'Demo Team',
      description: 'A demo team for testing purposes',
      ownerId: adminUser.id,
    },
  });

  console.log('✅ Created demo team:', demoTeam.name);

  // Add demo user to team
  await prisma.teamMember.upsert({
    where: {
      userId_teamId: {
        userId: demoUser.id,
        teamId: demoTeam.id,
      },
    },
    update: {},
    create: {
      userId: demoUser.id,
      teamId: demoTeam.id,
      role: 'MEMBER',
    },
  });

  console.log('✅ Added demo user to team');

  // Create demo tasks
  const demoTasks = [
    {
      title: 'Complete project documentation',
      description: 'Write comprehensive documentation for the OptoPomodoro API',
      priority: 'HIGH',
      estimatedPomodoros: 3,
      assigneeId: demoUser.id,
      tags: '["documentation", "api", "urgent"]',
      status: 'IN_PROGRESS',
    },
    {
      title: 'Review pull requests',
      description: 'Review and approve pending pull requests',
      priority: 'MEDIUM',
      estimatedPomodoros: 2,
      assigneeId: demoUser.id,
      tags: '["development", "review"]',
      status: 'TODO',
    },
    {
      title: 'Team standup meeting',
      description: 'Daily standup with the development team',
      priority: 'LOW',
      estimatedPomodoros: 1,
      assigneeId: demoUser.id,
      tags: '["meeting", "team"]',
      status: 'COMPLETED',
      completedAt: new Date(),
    },
  ];

  for (const taskData of demoTasks) {
    const task = await prisma.task.upsert({
      where: { id: `task-${taskData.title.replace(/\s+/g, '-').toLowerCase()}` },
      update: {},
      create: {
        id: `task-${taskData.title.replace(/\s+/g, '-').toLowerCase()}`,
        ...taskData,
        completedPomodoros: taskData.status === 'COMPLETED' ? taskData.estimatedPomodoros : 0,
      },
    });
    console.log('✅ Created demo task:', task.title);
  }

  // Create demo sessions
  const demoSessions = [
    {
      type: 'POMODORO',
      duration: 25,
      quality: 4,
      notes: 'Great focus session on documentation',
      completed: true,
      startTime: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
      endTime: new Date(Date.now() - 2 * 60 * 60 * 1000 + 25 * 60 * 1000),
    },
    {
      type: 'SHORT_BREAK',
      duration: 5,
      completed: true,
      startTime: new Date(Date.now() - 2 * 60 * 60 * 1000 + 25 * 60 * 1000),
      endTime: new Date(Date.now() - 2 * 60 * 60 * 1000 + 30 * 60 * 1000),
    },
    {
      type: 'POMODORO',
      duration: 25,
      quality: 5,
      notes: 'Excellent session, completed documentation section',
      completed: true,
      startTime: new Date(Date.now() - 1 * 60 * 60 * 1000), // 1 hour ago
      endTime: new Date(Date.now() - 1 * 60 * 60 * 1000 + 25 * 60 * 1000),
    },
  ];

  for (const sessionData of demoSessions) {
    const session = await prisma.session.create({
      data: {
        ...sessionData,
        userId: demoUser.id,
      },
    });
    console.log('✅ Created demo session:', session.type);
  }

  // Create achievements
  const achievements = [
    {
      name: 'First Pomodoro',
      description: 'Complete your first Pomodoro session',
      icon: '🍅',
      category: 'FOCUS',
      requirementType: 'SESSION_COUNT',
      requirementValue: 1,
      xpReward: 10,
    },
    {
      name: 'Week Warrior',
      description: 'Complete 25 Pomodoro sessions in a week',
      icon: '⚔️',
      category: 'CONSISTENCY',
      requirementType: 'SESSION_COUNT',
      requirementValue: 25,
      requirementTimeframe: 'WEEKLY',
      xpReward: 50,
    },
    {
      name: 'Focus Master',
      description: 'Complete 100 Pomodoro sessions',
      icon: '🧘',
      category: 'FOCUS',
      requirementType: 'SESSION_COUNT',
      requirementValue: 100,
      xpReward: 100,
    },
    {
      name: 'Team Player',
      description: 'Join a team and complete a task together',
      icon: '🤝',
      category: 'COLLABORATION',
      requirementType: 'TEAM_HELP',
      requirementValue: 1,
      xpReward: 25,
    },
  ];

  for (const achievementData of achievements) {
    const achievement = await prisma.achievement.upsert({
      where: { name: achievementData.name },
      update: {},
      create: achievementData,
    });
    console.log('✅ Created achievement:', achievement.name);
  }

  // Unlock some achievements for demo user
  const userAchievements = [
    'First Pomodoro',
    'Team Player',
  ];

  for (const achievementName of userAchievements) {
    const achievement = await prisma.achievement.findUnique({
      where: { name: achievementName },
    });

    if (achievement) {
      await prisma.userAchievement.upsert({
        where: {
          userId_achievementId: {
            userId: demoUser.id,
            achievementId: achievement.id,
          },
        },
        update: {},
        create: {
          userId: demoUser.id,
          achievementId: achievement.id,
          progress: 100,
        },
      });
      console.log('✅ Unlocked achievement for demo user:', achievementName);
    }
  }

  // Create demo challenge
  const demoChallenge = await prisma.challenge.create({
    data: {
      name: 'Weekly Focus Challenge',
      description: 'Complete 50 Pomodoro sessions this week',
      type: 'FOCUS_TIME',
      targetValue: 50,
      currentValue: 12,
      startDate: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000), // 3 days ago
      endDate: new Date(Date.now() + 4 * 24 * 60 * 60 * 1000), // 4 days from now
      createdBy: adminUser.id,
      teamId: demoTeam.id,
      rewardXp: 100,
      rewardBadge: 'Weekly Champion',
    },
  });

  console.log('✅ Created demo challenge:', demoChallenge.name);

  // Add participants to challenge
  await prisma.user.update({
    where: { id: demoUser.id },
    data: {
      challengeParticipants: {
        connect: { id: demoChallenge.id },
      },
    },
  });

  console.log('✅ Added demo user to challenge');

  console.log('\n🎉 Database seeding completed successfully!');
  console.log('\n📋 Demo Accounts:');
  console.log('   Admin: admin@optopomodoro.com / admin123');
  console.log('   Demo:  demo@optopomodoro.com  / demo123');
  console.log('\n🌐 Available at: http://localhost:3001/api/docs');
}

main()
  .catch((e) => {
    console.error('❌ Error during seeding:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });