 @ProductDesign.md @UIUX.md  you are a Tech Lead that has years of experience in making fully-detailed plan for
  development teams that you are leading. here the product design and the UI and UX that we must follow. the app
  is Product-ready so we need minimal changes for delivering the needed features. \
  The feature we need to deliver is dashboard screen, you will find the UIUX design that we must strict with. \
  do not divert from the provided UI and UX. each plan workflow MUST be according to existed proof, do not guess
  or fake any data. you can ask me for anything that are vauge, needs discussion or the workflow. the plan must be
  with main needed implementation as for backend: The needed APIs and the response example that will be according
  to @packages/backend/prisma/schema.prisma, as for front-end: the needed components, UI Flow that matches the
  UI/UX. ultrathink  



  ---------------------

  I need you to re-evaluate the plan with the backend for comparing between the actual implementation with the expected types to prevent
  missalignment ultrathink 



  -----------------------


   make the priority for fix these issues: Issue #1: Team Member Completion Rate

    Location: analytics.service.ts:255
    completionRate: 0,  // HARDCODED - NEEDS REAL CALCULATION
    Problem: All team members show 0% completion rate
    Impact: Team analytics dashboard will show incorrect data

    Issue #2: Mock Wellness Data

    Location: analytics.service.ts:136, 138
    hydrationCurrent: Math.round(Math.random() * 8),  // RANDOM DATA
    movementCurrent: Math.round(Math.random() * 5),  // RANDOM DATA
    Problem: Hydration and movement metrics use random data
    Impact: Wellness dashboard shows fake data

    Issue #3: Missing Analytics DTOs

    Problem: No request/response DTOs for type safety
    Impact: Runtime type errors possible 




    ----------------------------------------------

    > create folder called project_features_plans and add the agreed plan, 






    -----------------------------------------------

     as you are senior backend engineer experienced in NestJS, there are fixes that needed to be implemented in the @packages/backend/src/analytics/
  service, the acutal plan is @project_features_plans/01-backend-analytics-fixes.md with detailed info and detailed fixes, you need to check also
  @packages/backend/prisma/schema.prisma  for more insights. ultrathink



  -----------------------------------------------------


  as you are senior frontend engineer experienced in React, there is new feature needs to be implemented, the
  dashboard feature,  the acutal plan is in @project_features_plans/02-dashboard-implementation.md  with detailed
  info and needed implementation. ultrathink



  --------------------------------------------

  the backend is already running, start testing the
  endpoints.  