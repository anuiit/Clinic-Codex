import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from '../App'

vi.mock('../pages/WorkspacePage', () => ({
  default: () => <div data-testid="workspace-page">Workspace route</div>,
}))

vi.mock('../pages/AnnotationPage', () => ({
  default: () => <div data-testid="annotation-page">Annotation route</div>,
}))

function renderAt(path: string) {
  window.history.pushState({}, '', path)
  return render(<App />)
}

afterEach(() => {
  window.history.pushState({}, '', '/')
})

describe('App route contracts', () => {
  it('renders the shared app shell and workspace route at /', () => {
    const { container } = renderAt('/')

    expect(container.querySelector('main')).not.toBeNull()
    expect(screen.getByTestId('workspace-page')).toBeInTheDocument()
  })

  it('keeps the annotation page on the separate /annotate/:id route', () => {
    renderAt('/annotate/alpha-run')

    expect(screen.getByTestId('annotation-page')).toBeInTheDocument()
    expect(screen.queryByTestId('workspace-page')).not.toBeInTheDocument()
  })

  it('redirects the legacy dashboard route to the workspace without merging pages', async () => {
    renderAt('/dashboard')

    await waitFor(() => expect(window.location.pathname).toBe('/'))
    expect(screen.getByTestId('workspace-page')).toBeInTheDocument()
  })

  it('preserves legacy /analysis/:id handoff as a workspace query redirect', async () => {
    renderAt('/analysis/alpha run')

    await waitFor(() => {
      expect(window.location.pathname).toBe('/')
      expect(window.location.search).toBe('?analysis=alpha%20run')
    })
    expect(screen.getByTestId('workspace-page')).toBeInTheDocument()
  })
})
