const metrics = [
  { label: 'Sensor nodes', value: '4', detail: 'ESP32-S3 CSI anchors' },
  { label: 'Compute target', value: 'GH200', detail: 'Helios Slurm training' },
  { label: 'Pipeline layers', value: '3', detail: 'Bronze, Silver, Gold' },
  { label: 'Serving modes', value: '2', detail: 'Edge ONNX and cloud Triton' },
]

const layers = [
  {
    title: 'Firmware and RF capture',
    eyebrow: 'Edge layer',
    body: 'ESP32-S3 nodes encode WiFi CSI into CRC-protected binary packets, stream over UDP, and keep a stable device identity for signed OTA rollouts.',
  },
  {
    title: 'Gateway data plane',
    eyebrow: 'Ingestion layer',
    body: 'A Rust/Tokio gateway validates packets, estimates node health, buffers locally in SQLite, publishes to NATS, and uploads immutable Bronze batches to object storage.',
  },
  {
    title: 'Research data lake',
    eyebrow: 'Dataset layer',
    body: 'Dagster, Polars, PyArrow, and quality checks transform raw CSI into versioned Silver and Gold datasets with manifests, statistics, and normalization contracts.',
  },
  {
    title: 'Training and deployment',
    eyebrow: 'Model layer',
    body: 'PyTorch, Hydra, MLflow, LoRA adapters, knowledge distillation, ONNX export, model cards, and Helios GH200 Slurm jobs form the full MLOps loop.',
  },
]

const capabilities = [
  'Four-node room sensing topology',
  'CRC-checked CSI packet contract',
  'NATS JetStream upstream transport',
  'S3-compatible Bronze archive',
  'Versioned dataset registry',
  'LoRA room adaptation',
  'Knowledge distillation path',
  'Triton and ONNX serving contracts',
  'Prometheus, Grafana, Loki, OpenTelemetry',
  'mTLS, signed OTA, SOPS/Vault posture',
]

const stack = [
  ['Firmware', 'ESP-IDF C/C++'],
  ['Gateway', 'Rust, Tokio, SQLite'],
  ['Transport', 'UDP, NATS JetStream'],
  ['Control plane', 'FastAPI, PostgreSQL'],
  ['Lakehouse', 'MinIO/S3, Dagster'],
  ['ML', 'PyTorch, Hydra, MLflow'],
  ['HPC', 'Helios GH200 Slurm'],
  ['Serving', 'ONNX Runtime, Triton'],
]

export default function Home() {
  return (
    <main className="min-h-[100dvh] overflow-hidden bg-[#f7f8f5] text-zinc-950">
      <div className="pointer-events-none fixed inset-0 opacity-[0.04] [background-image:radial-gradient(#18181b_1px,transparent_1px)] [background-size:18px_18px]" />
      <section className="relative mx-auto grid min-h-[100dvh] max-w-[1400px] grid-cols-1 gap-10 px-5 py-6 md:grid-cols-[1.05fr_0.95fr] md:px-10 lg:px-14">
        <header className="flex items-center justify-between md:col-span-2">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-2xl border border-zinc-900/10 bg-white shadow-[0_18px_60px_-30px_rgba(24,24,27,0.35)]">
              <span className="h-3 w-3 rounded-full bg-emerald-500 shadow-[0_0_0_6px_rgba(16,185,129,0.12)]" />
            </div>
            <div>
              <p className="text-sm font-semibold tracking-tight">RF-WorldPose</p>
              <p className="text-xs text-zinc-500">Production research platform</p>
            </div>
          </div>
          <nav className="hidden items-center gap-7 text-sm text-zinc-600 md:flex">
            <a className="transition hover:text-zinc-950" href="/nodes">Nodes</a>
            <a className="transition hover:text-zinc-950" href="/datasets">Datasets</a>
            <a className="transition hover:text-zinc-950" href="/training">Training</a>
            <a className="transition hover:text-zinc-950" href="/models">Models</a>
          </nav>
        </header>

        <div className="flex flex-col justify-center pb-10 pt-12 md:pb-20 md:pt-10">
          <div className="mb-8 inline-flex w-fit items-center gap-3 rounded-full border border-zinc-900/10 bg-white/80 px-4 py-2 text-sm text-zinc-600 shadow-[0_20px_70px_-45px_rgba(24,24,27,0.45)] backdrop-blur">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            WiFi CSI sensing, from silicon to model registry
          </div>
          <h1 className="max-w-[780px] text-5xl font-semibold tracking-[-0.06em] text-zinc-950 md:text-7xl md:leading-[0.88]">
            A production-grade stack for RF human perception.
          </h1>
          <p className="mt-7 max-w-[650px] text-lg leading-8 text-zinc-600">
            RF-WorldPose turns four ESP32-S3 WiFi CSI nodes into a full sensing platform: edge capture, reliable ingestion, versioned data lakes, Helios GH200 training, and controlled edge or cloud inference.
          </p>
          <div className="mt-9 flex flex-col gap-3 sm:flex-row">
            <a className="rounded-full bg-zinc-950 px-6 py-3 text-center text-sm font-medium text-white transition active:translate-y-[1px]" href="/nodes">
              Open operations view
            </a>
            <a className="rounded-full border border-zinc-900/10 bg-white px-6 py-3 text-center text-sm font-medium text-zinc-800 transition hover:border-zinc-900/20 active:translate-y-[1px]" href="https://github.com/tientruongminh/rf-worldpose">
              View repository
            </a>
          </div>
        </div>

        <div className="relative flex items-center pb-12 md:pb-20">
          <div className="absolute -right-24 top-16 h-72 w-72 rounded-full bg-emerald-200/60 blur-3xl" />
          <div className="absolute bottom-12 left-0 h-64 w-64 rounded-full bg-zinc-300/60 blur-3xl" />
          <div className="relative w-full rounded-[2.5rem] border border-white/70 bg-white/80 p-4 shadow-[0_40px_110px_-55px_rgba(24,24,27,0.5)] backdrop-blur-xl">
            <div className="rounded-[2rem] border border-zinc-900/10 bg-[#10140f] p-5 text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.10)]">
              <div className="flex items-center justify-between border-b border-white/10 pb-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-emerald-300">Live pipeline</p>
                  <p className="mt-1 text-sm text-zinc-400">Room 01 reference deployment</p>
                </div>
                <div className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-200">Ready for bring-up</div>
              </div>
              <div className="mt-6 grid grid-cols-2 gap-3">
                {metrics.map((item) => (
                  <div key={item.label} className="rounded-3xl border border-white/10 bg-white/[0.04] p-4">
                    <p className="text-xs text-zinc-400">{item.label}</p>
                    <p className="mt-3 text-3xl font-semibold tracking-tight">{item.value}</p>
                    <p className="mt-2 text-xs leading-5 text-zinc-500">{item.detail}</p>
                  </div>
                ))}
              </div>
              <div className="mt-4 rounded-3xl border border-white/10 bg-white/[0.04] p-4">
                <div className="flex items-center justify-between text-xs text-zinc-400">
                  <span>CSI window</span>
                  <span>4 nodes x 60 subcarriers</span>
                </div>
                <div className="mt-5 grid h-40 grid-cols-24 items-end gap-1 overflow-hidden rounded-2xl bg-black/20 p-3">
                  {Array.from({ length: 48 }).map((_, i) => (
                    <span
                      key={i}
                      className="rounded-t bg-emerald-300/80"
                      style={{ height: `${24 + ((i * 19) % 96)}%`, opacity: 0.28 + ((i % 7) * 0.08) }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="relative mx-auto max-w-[1400px] px-5 pb-24 md:px-10 lg:px-14">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[0.82fr_1.18fr]">
          <div className="pt-4">
            <p className="text-sm font-medium uppercase tracking-[0.24em] text-emerald-700">Architecture</p>
            <h2 className="mt-4 max-w-[520px] text-4xl font-semibold tracking-[-0.045em] md:text-6xl md:leading-[0.95]">
              Built like infrastructure, not a demo notebook.
            </h2>
            <p className="mt-6 max-w-[560px] text-base leading-7 text-zinc-600">
              The repository separates hardware capture, data transport, lakehouse ETL, research training, model governance, serving, and operations. Each layer owns a contract that can be tested independently.
            </p>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {layers.map((layer, index) => (
              <article key={layer.title} className={`rounded-[2rem] border border-zinc-900/10 bg-white p-6 shadow-[0_28px_70px_-45px_rgba(24,24,27,0.45)] ${index % 2 === 1 ? 'md:translate-y-10' : ''}`}>
                <p className="text-xs font-medium uppercase tracking-[0.22em] text-emerald-700">{layer.eyebrow}</p>
                <h3 className="mt-5 text-2xl font-semibold tracking-tight text-zinc-950">{layer.title}</h3>
                <p className="mt-4 text-sm leading-7 text-zinc-600">{layer.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="relative border-y border-zinc-900/10 bg-white/70">
        <div className="mx-auto grid max-w-[1400px] grid-cols-1 gap-10 px-5 py-20 md:px-10 lg:grid-cols-[1.1fr_0.9fr] lg:px-14">
          <div>
            <p className="text-sm font-medium uppercase tracking-[0.24em] text-emerald-700">Capability map</p>
            <div className="mt-8 grid grid-cols-1 gap-x-8 gap-y-4 md:grid-cols-2">
              {capabilities.map((item) => (
                <div key={item} className="flex items-start gap-3 border-t border-zinc-900/10 pt-4">
                  <span className="mt-2 h-1.5 w-1.5 rounded-full bg-emerald-600" />
                  <p className="text-sm leading-6 text-zinc-700">{item}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-[2.5rem] bg-zinc-950 p-6 text-white shadow-[0_35px_90px_-50px_rgba(24,24,27,0.7)]">
            <div className="border-b border-white/10 pb-5">
              <p className="text-xs uppercase tracking-[0.24em] text-emerald-300">System stack</p>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight">Production contracts across the full path.</h2>
            </div>
            <div className="mt-3 divide-y divide-white/10">
              {stack.map(([k, v]) => (
                <div key={k} className="grid grid-cols-[0.65fr_1fr] gap-4 py-4 text-sm">
                  <span className="text-zinc-500">{k}</span>
                  <span className="text-zinc-100">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}
