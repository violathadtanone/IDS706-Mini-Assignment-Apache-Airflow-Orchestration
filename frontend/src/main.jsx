import React, {useEffect, useRef, useState} from 'react'
import {createRoot} from 'react-dom/client'
import {ReactFlow, Background, Controls, Handle, Position, MarkerType} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import './style.css'

const positions = {
  hospital_data_loader: {x: 0, y: 0}, cohort_builder: {x: 235, y: 0},
  score_model_trainer: {x: 470, y: 0}, error_reviewer: {x: 705, y: 0},
  regularization_adjuster: {x: 470, y: 210}, results_publisher: {x: 940, y: 0},
}
const format = value => value == null ? '—' : Number(value).toFixed(6)
const human = value => value?.replaceAll('_', ' ') ?? '—'

async function request(url, options) {
  const response = await fetch(url, options)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  return response.json()
}

function Agent({data}) {
  return <div className={`agent ${data.status}`}>
    <Handle type="target" position={Position.Left} id="in"/>
    <Handle type="source" position={Position.Right} id="out"/>
    <Handle type="source" position={Position.Bottom} id="down"/>
    <Handle type="target" position={Position.Bottom} id="return"/>
    <Handle type="source" position={Position.Top} id="up"/>
    <span className="agent-number">{data.number} / AGENT</span>
    <strong>{data.label}</strong>
    <span className="node-status"><i/>{data.status}</span>
  </div>
}
const nodeTypes = {agent: Agent}

function App() {
  const [graph, setGraph] = useState(null)
  const [run, setRun] = useState(null)
  const [starting, setStarting] = useState(false)
  const [flow, setFlow] = useState(null)
  const [error, setError] = useState('')
  const timer = useRef(null)
  useEffect(() => {
    request('/api/graph').then(setGraph).catch(e => setError(e.message))
    return () => clearTimeout(timer.current)
  }, [])
  useEffect(() => {
    if (!flow) return
    let resizeTimer
    const fit = () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => flow.fitView({padding: 0.12}), 100) }
    window.addEventListener('resize', fit)
    return () => { window.removeEventListener('resize', fit); clearTimeout(resizeTimer) }
  }, [flow])

  async function poll(id) {
    try {
      const next = await request(`/api/runs/${id}`)
      setRun(next)
      if (next.status === 'running') timer.current = setTimeout(() => poll(id), 400)
    } catch (e) { setError(e.message) }
  }
  async function start() {
    clearTimeout(timer.current)
    setStarting(true); setError(''); setRun(null)
    try {
      const created = await request('/api/runs', {method: 'POST'})
      await poll(created.id)
    } catch (e) { setError(e.message) }
    finally { setStarting(false) }
  }
  const busy = starting || run?.status === 'running'
  const result = run?.result
  const cohort = result || run?.cohort
  const attempts = run?.attempts ?? []
  const nodes = (graph?.nodes ?? []).map((n, index) => ({
    id: n.id, position: positions[n.id], type: 'agent',
    data: {...n, number: String(index + 1).padStart(2, '0'), status: run?.nodes[n.id] ?? 'waiting'},
  }))
  const observed = new Set()
  const starts = (run?.events ?? []).filter(e => e.status === 'running')
  starts.forEach((event, index) => {
    if (index) observed.add(`${starts[index - 1].node}->${event.node}`)
  })
  const edges = (graph?.edges ?? []).map(e => {
    const back = e.source === 'regularization_adjuster'
    const revise = e.target === 'regularization_adjuster'
    const traveled = observed.has(`${e.source}->${e.target}`)
    const active = run?.active_node === e.target && starts.at(-2)?.node === e.source
    return {...e, id: `${e.source}->${e.target}`, type: back ? 'default' : 'smoothstep',
      sourceHandle: back ? 'up' : revise ? 'down' : 'out',
      targetHandle: back ? 'return' : 'in',
      label: back ? 'Next alpha · train again' : revise ? 'Another attempt' : e.target === 'results_publisher' ? 'Finish' : undefined,
      animated: active,
      markerEnd: {type: MarkerType.ArrowClosed, color: traveled ? '#087e8b' : '#a4b4c1'},
      style: {stroke: traveled ? '#087e8b' : '#a4b4c1', strokeWidth: traveled ? 3 : 1.5},
      labelStyle: {fontSize: 11, fill: '#345369'}, labelBgPadding: [7, 4],
    }
  })
  const label = id => graph?.nodes.find(n => n.id === id)?.label ?? id
  return <main>
    <header><a className="brand" href="/">H<span>+</span> <b>HOSPITAL SIGNALS</b></a><span className="eyebrow">IDS706 / HEALTHCARE ML</span></header>
    <section className="intro"><div><p className="eyebrow">PART 2 · LIVE AGENT WORKFLOW</p><h1>From hospital attributes<br/>to infection-score estimates.</h1><p className="description">Six deterministic agents. One bounded Ridge regression loop.<br/>A retrospective association study using processed PostgreSQL data.</p></div>
      <div className="run-control"><button onClick={start} disabled={busy || !graph}>{busy ? 'Workflow running…' : 'Start workflow run'}<span>↗</span></button><p>HAI_1_SIR · central-line bloodstream infection score</p><span className={`badge ${run?.status ?? ''}`}>{run?.status ?? 'Ready to run'}</span></div></section>
    {(error || run?.error) && <div role="alert" className="error">{error || run.error}<button onClick={() => run && poll(run.id)}>Reconnect to run</button></div>}
    <section className="panel graph-panel"><div className="panel-heading"><div><h2>Live agent graph</h2><p>Statuses come from actual node execution. Teal edges show the path taken.</p></div><div className="training-status" aria-live="polite">{run?.active_node ? <><b>{label(run.active_node)}</b><span>Attempt {run.attempt || '—'} · Ridge α {run.alpha ?? '—'}</span></> : <><b>{result ? 'Run complete' : run?.status === 'failed' ? 'Run failed' : 'Awaiting execution'}</b><span>{attempts.length} training attempts recorded</span></>}</div></div>
      <div className="graph"><ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onInit={setFlow} fitView fitViewOptions={{padding: 0.12}} nodesDraggable={false} nodesConnectable={false} minZoom={0.15} maxZoom={1.5}><Background gap={22} color="#d8e2e8"/><Controls showInteractive={false}/></ReactFlow></div>
      <div className="legend"><span>○ Waiting</span><span>◉ Running</span><span>● Completed</span><span>× Failed</span><span className="loop-note">Regularization Adjuster → Score Model Trainer is the tuning back edge.</span></div>
    </section>
    <section className="metrics" aria-label="Evaluation metrics">
      <Metric title="Cohort" value={cohort?.facilities?.toLocaleString() ?? '—'} detail="Unique facilities"/>
      <Metric title="Train / validation / test" value={cohort?.split_sizes ? Object.values(cohort.split_sizes).join(' / ') : '—'} detail={cohort ? `${cohort.start_date} → ${cohort.end_date}` : 'Fixed facility-disjoint splits'}/>
      <Metric title="Baseline validation MAE" value={format(result?.baseline_validation_mae ?? run?.baseline_validation_mae)} detail="Training-median predictor · lower is better"/>
    </section>
    <div className="columns"><section className="panel"><div className="panel-heading"><div><h2>Regularization attempts</h2><p>Validation evidence drives every adjustment.</p></div></div>
      <table><thead><tr><th>Attempt</th><th>Ridge α</th><th>Train MAE</th><th>Validation MAE</th></tr></thead><tbody>{attempts.map((a, i) => <tr key={a.alpha}><td>{String(i + 1).padStart(2, '0')}</td><td>{a.alpha}</td><td>{format(a.train_mae)}</td><td>{format(a.validation_mae)}</td></tr>)}</tbody></table>{!attempts.length && <p className="empty">Training results will appear here during execution.</p>}
    </section><section className="panel results"><div className="panel-heading"><div><h2>Selected predictor</h2><p>Selected by validation MAE; test data stays held out until tuning ends.</p></div></div>
      <h3>{result ? result.selected_candidate === 'median_baseline' ? 'Median baseline selected' : `Ridge selected · α ${result.best_ridge_alpha}` : 'Selection pending'}</h3>
      <p>{result?.selected_candidate === 'median_baseline' ? 'No Ridge attempt improved on the baseline validation MAE.' : 'Final evaluation appears after tuning is complete.'}</p>
      <div className="result-metrics"><Metric title="Final test MAE" value={format(result?.test_mae)}/><Metric title="Baseline test MAE" value={format(result?.baseline_test_mae)}/></div>
      <div className="stop"><span>STOP REASON</span><strong>{human(result?.stop_reason)}</strong></div>
    </section></div>
    <section className="panel events"><div className="panel-heading"><div><h2>Execution record</h2><p>Every start and finish is retained, including nodes too fast to see live.</p></div><span className="eyebrow">{run?.events?.length ?? 0} EVENTS</span></div>
      <ol>{(run?.events ?? []).map(e => <li key={e.sequence}><time>{new Date(e.timestamp).toLocaleTimeString()}</time><strong>{label(e.node)}</strong><span className={e.status}>{e.status}</span>{e.node === 'score_model_trainer' && <span>Attempt {e.attempt} · α {e.alpha}</span>}</li>)}</ol>{!run?.events?.length && <p className="empty">Start a run to follow the agents.</p>}
    </section><footer>MAE = mean absolute error, in infection-score units. Lower is better.<br/>Hospital ratings may reflect related safety information. This is not clinical forecasting.</footer>
  </main>
}
function Metric({title, value, detail}) {return <article className="metric"><span>{title}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</article>}
createRoot(document.getElementById('root')).render(<App/> )
