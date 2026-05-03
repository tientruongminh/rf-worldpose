const cards = [
  ['Nodes online', '4 / 4'], ['Packet drop', '< 1%'], ['Model', 'rfworldpose candidate'], ['Training jobs', '0 running']
]
export default function Home() {
  return <main style={{padding:32,fontFamily:'Inter, sans-serif',background:'#080b12',color:'#e8eefc',minHeight:'100vh'}}>
    <h1>RF-WorldPose Dashboard</h1><p>Live RF sensing operations: nodes, CSI, skeleton, datasets, jobs, models, alerts.</p>
    <section style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:16}}>{cards.map(([k,v])=><div key={k} style={{border:'1px solid #26334d',borderRadius:16,padding:20,background:'#101827'}}><div style={{color:'#8aa1c7'}}>{k}</div><strong style={{fontSize:28}}>{v}</strong></div>)}</section>
    <section style={{marginTop:24,display:'grid',gridTemplateColumns:'2fr 1fr',gap:16}}><div style={{height:360,border:'1px solid #26334d',borderRadius:16,padding:20}}>Live skeleton / Three.js canvas placeholder</div><div style={{border:'1px solid #26334d',borderRadius:16,padding:20}}>Alerts<br/>No active alerts</div></section>
  </main>
}
