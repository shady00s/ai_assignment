import { applyDecorators, Type } from '@nestjs/common';
import { ApiExtraModels, ApiResponse, getSchemaPath } from '@nestjs/swagger';

/**
 * Creates a standardized API response wrapper
 * Matches frontend expectations with { success: boolean, data: T }
 */
export const ApiStandardResponse = <TModel extends Type<any>>(
  model: TModel,
  description?: string,
) => {
  return applyDecorators(
    ApiExtraModels(model),
    ApiResponse({
      status: 200,
      description: description || 'Successful operation',
      schema: {
        type: 'object',
        properties: {
          success: {
            type: 'boolean',
            example: true,
            description: 'Whether the operation was successful',
          },
          data: {
            $ref: getSchemaPath(model),
          },
        },
      },
    }),
  );
};

/**
 * Creates a standardized API response for arrays
 */
export const ApiStandardArrayResponse = <TModel extends Type<any>>(
  model: TModel,
  description?: string,
) => {
  return applyDecorators(
    ApiExtraModels(model),
    ApiResponse({
      status: 200,
      description: description || 'Successful operation',
      schema: {
        type: 'object',
        properties: {
          success: {
            type: 'boolean',
            example: true,
            description: 'Whether the operation was successful',
          },
          data: {
            type: 'array',
            items: {
              $ref: getSchemaPath(model),
            },
          },
        },
      },
    }),
  );
};

/**
 * Creates a response for DELETE operations
 */
export const ApiDeleteResponse = (description?: string) => {
  return ApiResponse({
    status: 200,
    description: description || 'Successfully deleted resource',
    schema: {
      type: 'object',
      properties: {
        success: {
          type: 'boolean',
          example: true,
          description: 'Whether the operation was successful',
        },
        data: {
          type: 'null',
          example: null,
        },
      },
    },
  });
};