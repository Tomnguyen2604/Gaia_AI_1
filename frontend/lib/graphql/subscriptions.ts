/**
 * GraphQL Subscriptions
 */

import { gql } from '@apollo/client';

export const CHAT_STREAM_SUBSCRIPTION = gql`
  subscription ChatStream($input: ChatInput!) {
    chatStream(input: $input) {
      token
      isComplete
      totalTokens
    }
  }
`;
