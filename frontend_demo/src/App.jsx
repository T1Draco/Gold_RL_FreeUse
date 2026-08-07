import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Activity, BrainCircuit } from 'lucide-react'; // Icono BrainCircuit para aprendizaje
import { useDataLoader } from './hooks/useDataLoader';
import { TradingChart } from './components/TradingChart';
import { DashboardStats } from './components/DashboardStats';

function App() {
  const { marketData, tradeEvents, totalSteps, initialStep, loading, initialBalance, firstTradeDate } = useDataLoader();
  
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(50);
  
  const [balance, setBalance] = useState(initialBalance);
  const [maxBalance, setMaxBalance] = useState(initialBalance);
  const [activeTrade, setActiveTrade] = useState(null);
  const [markers, setMarkers] = useState([]);

  const simulationRef = useRef();

  // Reset al inicio
  useEffect(() => {
    if (!loading) {
      setCurrentStep(0);
      setBalance(initialBalance);
    }
  }, [loading, initialBalance]);

  // Bucle Principal
  useEffect(() => {
    if (loading || !marketData[currentStep]) return;
    const todayStr = marketData[currentStep].time;
    const events = tradeEvents[todayStr] || [];

    events.forEach(e => {
      if (e.type === 'ENTRY') {
        setActiveTrade(e);
        setMarkers(prev => [...prev, { time: todayStr, type: 'ENTRY' }]);
      } else if (e.type === 'EXIT') {
        if (e.balance !== undefined) {
            setBalance(e.balance);
            setMaxBalance(prev => Math.max(prev, e.balance));
        }
        setActiveTrade(null);
        setMarkers(prev => [...prev, { time: todayStr, type: 'EXIT', pnl: e.pnl }]);
      }
    });

    if (isPlaying && currentStep < totalSteps) {
      simulationRef.current = setTimeout(() => setCurrentStep(p => p + 1), speedMs);
    } else setIsPlaying(false);

    return () => clearTimeout(simulationRef.current);
  }, [currentStep, isPlaying, loading]); 

  const reset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setBalance(initialBalance);
    setMaxBalance(initialBalance);
    setMarkers([]);
    setActiveTrade(null);
  };

  if (loading) return <div style={{height:'100vh', display:'flex', alignItems:'center', justifyContent:'center', color:'white'}}>Cargando...</div>;

  // Lógica de Estado: ¿Estamos entrenando o operando?
  const currentDate = marketData[currentStep]?.time;
  const isTrainingPhase = firstTradeDate && currentDate < firstTradeDate;

  return (
    <div className="app-container">
      
      {/* HEADER */}
      <header className="header">
        <div style={{display:'flex', gap:'15px', alignItems:'center'}}>
            <div style={{background: 'rgba(59, 130, 246, 0.2)', padding:'8px', borderRadius:'8px', display:'flex'}}>
                <Activity color="#3b82f6" size={20} />
            </div>
            <div className="title-box">
                <h1>Agente de Trading con RL</h1>
                <p>Deep Reinforcement Learning | XAU/USD</p>
            </div>
        </div>

        <div style={{display:'flex', gap:'20px', alignItems:'center'}}>
            <div style={{textAlign:'right'}}>
                <div style={{fontSize:'11px', color:'#9ca3af'}}>FECHA ACTUAL</div>
                <div style={{fontSize:'18px', fontWeight:'bold', fontFamily:'monospace'}}>{currentDate}</div>
            </div>
            
            {/* BADGE DE ESTADO DINÁMICO */}
            <div className={`status-badge ${isTrainingPhase ? '' : (activeTrade ? 'status-active' : '')}`} 
                 style={{ 
                     background: isTrainingPhase ? 'rgba(234, 179, 8, 0.1)' : undefined,
                     borderColor: isTrainingPhase ? 'rgba(234, 179, 8, 0.3)' : undefined,
                     color: isTrainingPhase ? '#eab308' : undefined
                 }}>
                
                {isTrainingPhase ? (
                    <>
                        <BrainCircuit size={14} />
                        <span>APRENDIENDO (WARMUP)</span>
                    </>
                ) : (
                    <>
                        <div style={{width:'8px', height:'8px', borderRadius:'50%', background: activeTrade ? '#10b981' : '#6b7280'}}></div>
                        <span>{activeTrade ? 'EN MERCADO' : 'BUSCANDO SEÑAL'}</span>
                    </>
                )}
            </div>
        </div>
      </header>

      <div className="main-layout">
        
        <aside className="sidebar">
            <div className="sidebar-section">
                <DashboardStats currentBalance={balance} initialBalance={initialBalance} maxBalanceSoFar={maxBalance} />
            </div>

            <div className="sidebar-section" style={{background: '#161e2e'}}>
                <div style={{display:'flex', gap:'10px', marginBottom:'15px'}}>
                    <button onClick={reset} className="btn btn-secondary" style={{width:'50px'}}>
                        <RotateCcw size={18} />
                    </button>
                    <button onClick={() => setIsPlaying(!isPlaying)} className="btn btn-primary">
                        {isPlaying ? <Pause size={18}/> : <Play size={18}/>} 
                        {isPlaying ? 'PAUSAR' : 'INICIAR'}
                    </button>
                </div>
                <div>
                    <div style={{display:'flex', justifyContent:'space-between', fontSize:'10px', color:'#6b7280', fontWeight:'bold', textTransform:'uppercase'}}>
                        <span>Lento</span><span>Velocidad</span><span>Rápido</span>
                    </div>
                    <input type="range" min="1" max="200" step="5" value={201 - speedMs} onChange={e => setSpeedMs(201 - Number(e.target.value))} />
                </div>
            </div>

            <div className="logs-panel">
                <div className="logs-header">
                    <span>Log de Operaciones</span>
                    <span>{markers.length}</span>
                </div>
                <div className="logs-list">
                    {markers.slice().reverse().map((m, i) => (
                        <div key={i} className="log-entry" style={{
                            borderLeftColor: m.type === 'ENTRY' ? '#eab308' : (m.pnl > 0 ? '#10b981' : '#ef4444')
                        }}>
                            <div style={{display:'flex', justifyContent:'space-between', marginBottom:'4px'}}>
                                <span style={{color:'#6b7280'}}>{m.time}</span>
                                <span className={m.type === 'ENTRY' ? 'text-yellow' : (m.pnl > 0 ? 'text-green' : 'text-red')}>
                                    {m.type === 'ENTRY' ? 'ENTRADA' : (m.pnl > 0 ? 'GANANCIA' : 'PÉRDIDA')}
                                </span>
                            </div>
                            {m.type === 'EXIT' && (
                                <div style={{textAlign:'right', fontWeight:'bold'}}>
                                    {m.pnl > 0 ? '+' : ''}{m.pnl?.toFixed(2)} USD
                                </div>
                            )}
                        </div>
                    ))}
                    {markers.length === 0 && (
                        <div style={{textAlign:'center', color:'#4b5563', marginTop:'40px', padding:'0 20px'}}>
                            {isTrainingPhase 
                                ? <div style={{fontStyle:'italic', opacity: 0.7}}>El agente está procesando datos históricos para calibrar sus pesos neuronales...<br/><br/>(1996 - 2001)</div>
                                : <div style={{fontStyle:'italic'}}>Esperando señal de mercado...</div>
                            }
                        </div>
                    )}
                </div>
            </div>
        </aside>

        <section className="chart-area">
             <TradingChart data={marketData} currentStep={currentStep} tradeMarkers={markers} />
             <div style={{position:'absolute', bottom:0, left:0, width:'100%', height:'4px', background:'#1f2937'}}>
                 <div style={{height:'100%', background: isTrainingPhase ? '#eab308' : '#3b82f6', width: `${(currentStep / totalSteps * 100)}%`, transition:'width 0.1s'}}></div>
             </div>
        </section>

      </div>
    </div>
  );
}

export default App;