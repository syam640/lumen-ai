import { useState, useCallback } from 'react'

const API_URL = import.meta.env.VITE_API_URL || ''

interface ScreenResult {
  success: boolean
  prediction: { stage: number; label: string; confidence: number }
  probabilities: Record<string, number>
  quality: { status: string; brightness: number; contrast: number; sharpness: number; width: number; height: number; message: string }
  triage: { priority: string; reason: string }
  explainability: { method: string; target_class: number; heatmap_available: boolean; visualization_url?: string; visualization_base64?: string }
  inference_time_ms: number
  error: string | null
}

const DR_LABELS: Record<number, string> = {
  0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'
}

const PRIORITY_COLORS: Record<string, string> = {
  LOW: '#22c55e', MODERATE: '#f59e0b', HIGH: '#ef4444', URGENT: '#7c2d12', UNKNOWN: '#94a3b8'
}

const QUALITY_COLORS: Record<string, string> = {
  GOOD: '#22c55e', ACCEPTABLE: '#f59e0b', POOR: '#ef4444', BAD: '#7c2d12'
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<ScreenResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFile = useCallback((f: File) => {
    setFile(f)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target?.result as string)
    reader.readAsDataURL(f)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [handleFile])

  const handleScreen = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API_URL}/api/v1/screen`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Screening failed')
      }
      setResult(await res.json())
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: 24 }}>
      <header style={{ textAlign: 'center', marginBottom: 32 }}>
        <h1 style={{ fontSize: 32, fontWeight: 700, color: '#1a1a2e', margin: 0 }}>LUMEN</h1>
        <p style={{ color: '#64748b', margin: '4px 0 0', fontSize: 14 }}>
          Explainable AI for Diabetic Retinopathy Screening
        </p>
        <p style={{ color: '#94a3b8', margin: '2px 0 0', fontSize: 12 }}>
          AI-assisted screening support for rural health workers
        </p>
      </header>

      {/* Upload */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => document.getElementById('file-input')?.click()}
        style={{
          border: '2px dashed #cbd5e1', borderRadius: 12, padding: 40,
          textAlign: 'center', cursor: 'pointer', background: '#fff',
          marginBottom: 24, transition: 'border-color 0.2s',
        }}
      >
        <input
          id="file-input" type="file" accept="image/*" hidden
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        {preview ? (
          <img src={preview} alt="Preview" style={{ maxHeight: 200, borderRadius: 8 }} />
        ) : (
          <p style={{ color: '#64748b' }}>Upload Fundus Image<br /><small>Drag/drop or click to select</small></p>
        )}
      </div>

      {file && !result && (
        <button
          onClick={handleScreen}
          disabled={loading}
          style={{
            width: '100%', padding: 14, fontSize: 16, fontWeight: 600,
            background: '#2563eb', color: '#fff', border: 'none',
            borderRadius: 8, cursor: loading ? 'wait' : 'pointer',
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? 'Screening in progress...' : 'Run DR Screening'}
        </button>
      )}

      {error && (
        <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, padding: 16, marginTop: 16, color: '#991b1b' }}>
          {error}
        </div>
      )}

      {result && result.success && (
        <div style={{ marginTop: 24 }}>
          {/* Quality */}
          <Card title="Image Quality">
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
              <StatusBadge label={result.quality.status} color={QUALITY_COLORS[result.quality.status] || '#94a3b8'} />
              <span style={{ color: '#64748b', fontSize: 13 }}>{result.quality.message}</span>
            </div>
            <div style={{ display: 'flex', gap: 24, fontSize: 13, color: '#475569' }}>
              <span>Brightness: {result.quality.brightness}</span>
              <span>Contrast: {result.quality.contrast}</span>
              <span>Sharpness: {result.quality.sharpness}</span>
            </div>
          </Card>

          {/* Prediction */}
          <Card title="AI Screening Result">
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 12 }}>
              <span style={{ fontSize: 28, fontWeight: 700 }}>
                Stage {result.prediction.stage}
              </span>
              <span style={{ fontSize: 18, color: '#475569' }}>
                {result.prediction.label}
              </span>
              <span style={{ fontSize: 14, color: '#94a3b8' }}>
                ({(result.prediction.confidence * 100).toFixed(1)}% confidence)
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {Object.entries(result.probabilities).map(([k, v]) => {
                const stage = parseInt(k)
                const isPred = stage === result.prediction.stage
                return (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                    <span style={{ width: 100, fontWeight: isPred ? 700 : 400 }}>
                      {stage} — {DR_LABELS[stage]}
                    </span>
                    <div style={{ flex: 1, background: '#e2e8f0', borderRadius: 4, height: 12 }}>
                      <div style={{
                        width: `${v * 100}%`, height: '100%',
                        background: isPred ? '#2563eb' : '#94a3b8',
                        borderRadius: 4,
                      }} />
                    </div>
                    <span style={{ width: 50, textAlign: 'right', fontSize: 12 }}>
                      {(v * 100).toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </Card>

          {/* Grad-CAM */}
          {result.explainability.heatmap_available && (result.explainability.visualization_base64 || result.explainability.visualization_url) && (
            <Card title="Explainability">
              <p style={{ fontSize: 12, color: '#94a3b8', margin: '0 0 8px' }}>
                Model-associated regions (Grad-CAM)
              </p>
              <img
                src={result.explainability.visualization_base64
                  || `${API_URL}${result.explainability.visualization_url}`}
                alt="Grad-CAM"
                style={{ width: '100%', borderRadius: 8, border: '1px solid #e2e8f0' }}
              />
            </Card>
          )}

          {/* Triage */}
          <Card title="Triage">
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <StatusBadge
                label={result.triage.priority}
                color={PRIORITY_COLORS[result.triage.priority] || '#94a3b8'}
              />
              <span style={{ fontSize: 13, color: '#475569' }}>{result.triage.reason}</span>
            </div>
          </Card>

          <p style={{ fontSize: 11, color: '#94a3b8', textAlign: 'center', marginTop: 16 }}>
            Inference: {result.inference_time_ms.toFixed(0)}ms
          </p>
        </div>
      )}

      <footer style={{ textAlign: 'center', marginTop: 40, padding: '20px 0', borderTop: '1px solid #e2e8f0' }}>
        <p style={{ fontSize: 11, color: '#94a3b8', maxWidth: 600, margin: '0 auto', lineHeight: 1.5 }}>
          LUMEN is an AI-assisted research prototype for screening support. It does not
          provide a definitive diagnosis and does not replace evaluation by a qualified
          eye-care professional.
        </p>
      </footer>
    </div>
  )
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
    }}>
      <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#1a1a2e' }}>
        {title}
      </h3>
      {children}
    </div>
  )
}

function StatusBadge({ label, color }: { label: string; color: string }) {
  return (
    <span style={{
      display: 'inline-block', padding: '4px 12px', borderRadius: 20,
      fontSize: 12, fontWeight: 600, color: '#fff', background: color,
    }}>
      {label}
    </span>
  )
}
