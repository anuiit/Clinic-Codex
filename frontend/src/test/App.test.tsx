import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from '../App'

describe('App', () => {
  it('renders without crashing and mounts a root element', () => {
    const { container } = render(<App />)
    expect(container.firstChild).not.toBeNull()
  })

  it('renders a main landmark element', () => {
    const { container } = render(<App />)
    expect(container.querySelector('main')).not.toBeNull()
  })
})
