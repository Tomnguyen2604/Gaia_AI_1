/**
 * GraphQL Mutations
 */

import { gql } from '@apollo/client';

export const CHAT_MUTATION = gql`
  mutation Chat($input: ChatInput!) {
    chat(input: $input) {
      response
      tokensGenerated
      generationTime
    }
  }
`;
