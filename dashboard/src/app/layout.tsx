import './globals.css'

export const metadata = {
  title: 'RF-WorldPose',
  description: 'Production research platform for WiFi CSI sensing and RF human perception.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
