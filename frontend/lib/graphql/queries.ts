/**
 * GraphQL Queries
 */

import { gql } from '@apollo/client';

export const HEALTH_QUERY = gql`
  query Health {
    health {
      status
      modelLoaded
      ragEnabled
      ragDocuments
      device
      cudaAvailable
    }
  }
`;

export const MODEL_INFO_QUERY = gql`
  query ModelInfo {
    modelInfo {
      name
      version
      contextWindow
      maxTokens
    }
  }
`;
