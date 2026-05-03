export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8080'
export async function getJson(path: string) { const r = await fetch(`${API_BASE}${path}`, { cache: 'no-store' }); if (!r.ok) throw new Error(`API ${r.status}`); return r.json() }
