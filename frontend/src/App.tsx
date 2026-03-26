import React, { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<
    'idle' | 'uploading' | 'processing' | 'done' | 'error'
  >('idle')
  const [error, setError] = useState<string | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    if (!file) {
      setError('Please select a video file.')
      return
    }

    if (resultUrl) URL.revokeObjectURL(resultUrl)
    setResultUrl(null)
    setStatus('uploading')

    const fd = new FormData()
    fd.append('video', file)

    try {
      setStatus('processing')
      const res = await fetch('http://localhost:8000/api/pose-overlay', {
        method: 'POST',
        body: fd,
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || `Request failed (${res.status})`)
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setResultUrl(url)
      setStatus('done')
    } catch (err: unknown) {
      setStatus('error')
      setError(err instanceof Error ? err.message : 'Unknown error')
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div className="brand">
          <div className="logo" aria-hidden="true" />
          <div>
            <div className="title">GarmentIQ</div>
            <div className="subtitle">Video → keypoints overlay (Body23)</div>
          </div>
        </div>
        <div className="pill">FastAPI + React</div>
      </header>

      <main className="grid">
        <section className="card">
          <h1>Upload a video</h1>
          <p className="muted">
            We’ll extract frames, run pose detection, and return an annotated mp4.
          </p>

          <form onSubmit={onSubmit} className="form">
            <label className="file">
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <div className="fileBody">
                <div className="fileName">
                  {file ? file.name : 'Choose a video file (mp4/mov/avi/mkv)'}
                </div>
                <div className="fileHint">Max size depends on your machine</div>
              </div>
            </label>

            <button className="btn" type="submit" disabled={!file || status === 'processing'}>
              {status === 'processing' ? 'Processing…' : 'Generate overlay'}
            </button>

            {error ? <div className="error">{error}</div> : null}
            <div className="status">
              Status: <strong>{status}</strong>
            </div>
          </form>
        </section>

        <section className="card">
          <h2>Result</h2>
          <p className="muted">Preview and download the processed video.</p>
          {resultUrl ? (
            <>
              <video className="video" controls src={resultUrl} />
              <a className="btn secondary" href={resultUrl} download="overlay.mp4">
                Download mp4
              </a>
            </>
          ) : (
            <div className="empty">No output yet.</div>
          )}
        </section>
      </main>

      <footer className="footer">
        Run backend on <code>localhost:8000</code> and frontend on <code>localhost:5173</code>.
      </footer>
    </div>
  )
}

export default App
