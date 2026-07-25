import { renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { useRuntimeMessageRepository } from './runtime-repository'

function user(id: string, text: string): ChatMessage {
  return { id, role: 'user', parts: [{ type: 'text', text }] }
}

function assistant(id: string, text: string): ChatMessage {
  return { id, role: 'assistant', parts: [{ type: 'text', text }] }
}

describe('useRuntimeMessageRepository', () => {
  it('exports every message in order', () => {
    const messages = [user('u1', 'hello'), assistant('a1', 'hi'), user('u2', 'again')]
    const { result } = renderHook(() => useRuntimeMessageRepository(messages))

    expect(result.current.messages.map(({ message }) => message.id)).toEqual(['u1', 'a1', 'u2'])
    expect(result.current.headId).toBe('u2')
  })

  // assistant-ui's MessageRepository throws on a repeated id ("a message with
  // the same id already exists in the parent tree"), and it throws during
  // render — so one duplicated row anywhere upstream takes the whole workspace
  // pane down with an error boundary. Dropping the repeat is always the better
  // failure: the transcript loses one row instead of the chat losing the pane.
  it('drops a repeated id rather than handing assistant-ui a duplicate', () => {
    const messages = [user('u1', 'hello'), assistant('a1', 'hi'), assistant('a1', 'hi again')]
    const { result } = renderHook(() => useRuntimeMessageRepository(messages))

    expect(result.current.messages.map(({ message }) => message.id)).toEqual(['u1', 'a1'])
  })

  it('keeps the first row under a repeated id and parents the rest off it', () => {
    const messages = [user('u1', 'hello'), assistant('a1', 'first'), user('u1', 'repeat'), assistant('a2', 'second')]
    const { result } = renderHook(() => useRuntimeMessageRepository(messages))

    expect(result.current.messages.map(({ message }) => message.id)).toEqual(['u1', 'a1', 'a2'])
    expect(result.current.messages.find(({ message }) => message.id === 'a2')?.parentId).toBe('a1')
    expect(result.current.headId).toBe('a2')
  })
})
