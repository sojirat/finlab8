'use client';

import { useMemo, useState } from 'react';

type Source = {
  sid: string;
  source?: string;
  page?: number;
  chunk?: number;
  preview?: string;
};

export default function Home() {
  const [query, setQuery] = useState('นายสมชายควรผ่านการอนุมัติหรือไม่? ขอเหตุผลและอ้างอิงหลักฐาน');
  const [result, setResult] = useState<string>('');
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const apiBase = useMemo(() => process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000', []);

  const handleAnalyze = async () => {
    setLoading(true);
    setResult('');
    setSources([]);
    try {
      const res = await fetch(`${apiBase}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query }),
      });

      const data = await res.json();
      if (!res.ok) {
        setResult(`ERROR: ${data?.detail ?? 'unknown'}`);
      } else {
        setResult(data.answer ?? '');
        setSources(data.sources ?? []);
      }
    } catch (e: any) {
      setResult(`Connection error: ${String(e?.message ?? e)}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 980, margin: '0 auto' }}>
      <h1 style={{ fontSize: 26, margin: 0, fontWeight: 700 }}>AI Underwriter Assistant (Lab 8)</h1>
      <p style={{ marginTop: 8, color: '#555' }}>
        RAG Demo: Ingest → Retrieve → Generate (ตอบพร้อมหลักฐาน [S#])
      </p>

      <label style={{ display: 'block', marginTop: 16, fontWeight: 600 }}>คำถาม</label>
      <textarea
        style={{ width: '100%', minHeight: 90, padding: 12, border: '1px solid #ddd', borderRadius: 8 }}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="ใส่คำถามวิเคราะห์ความเสี่ยง..."
      />

      <div style={{ display: 'flex', gap: 12, marginTop: 12 }}>
        <button
          onClick={handleAnalyze}
          disabled={loading}
          style={{
            background: loading ? '#999' : '#2563eb',
            color: '#fff',
            padding: '10px 14px',
            border: 'none',
            borderRadius: 8,
            cursor: loading ? 'not-allowed' : 'pointer',
            fontWeight: 700,
          }}
        >
          {loading ? 'กำลังวิเคราะห์...' : 'วิเคราะห์ความเสี่ยง'}
        </button>
        <a href={`${apiBase}/docs`} target="_blank" rel="noreferrer" style={{ padding: '10px 0' }}>
          เปิด API Docs
        </a>
      </div>

      <div style={{ marginTop: 18, padding: 14, border: '1px solid #eee', borderRadius: 10, background: '#fafafa' }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>ผลลัพธ์</div>
        <pre style={{ whiteSpace: 'pre-wrap', margin: 0, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}>
          {result || '—'}
        </pre>
      </div>

      <div style={{ marginTop: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>หลักฐานที่ถูกดึงมา (Sources)</div>
        {sources.length === 0 ? (
          <div style={{ color: '#777' }}>—</div>
        ) : (
          <div style={{ display: 'grid', gap: 10 }}>
            {sources.map((s) => (
              <div key={s.sid} style={{ border: '1px solid #eee', borderRadius: 10, padding: 12 }}>
                <div style={{ fontWeight: 700 }}>
                  [{s.sid}] {s.source ?? 'unknown'} {typeof s.page === 'number' ? `(page ${s.page})` : ''}{' '}
                  {typeof s.chunk === 'number' ? `(chunk ${s.chunk})` : ''}
                </div>
                <div style={{ color: '#555', marginTop: 6 }}>{s.preview ?? ''}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <hr style={{ margin: '20px 0', border: 'none', borderTop: '1px solid #eee' }} />

      <div style={{ color: '#666', fontSize: 13 }}>
        Tip: ถ้าต้องการเปลี่ยน API base ให้ตั้งค่า <code>NEXT_PUBLIC_API_BASE</code>.
      </div>
    </div>
  );
}
