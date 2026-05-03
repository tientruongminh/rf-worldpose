import { getJson } from '../lib'
export default async function Nodes(){ let data:any={nodes:[]}; try{data=await getJson('/api/v1/deployments/room01/status')}catch{} return <main style={{padding:32,fontFamily:'Inter, sans-serif'}}><h1>Nodes</h1><pre>{JSON.stringify(data.nodes||[],null,2)}</pre></main>}
