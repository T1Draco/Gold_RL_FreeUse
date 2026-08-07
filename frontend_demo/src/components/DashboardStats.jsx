import React from 'react';
import { DollarSign, AlertTriangle } from 'lucide-react';

export const DashboardStats = ({ currentBalance, initialBalance, maxBalanceSoFar }) => {
  const pnl = currentBalance - initialBalance;
  const pnlPercentage = (pnl / initialBalance) * 100;
  const isProfit = pnl >= 0;

  const currentDrawdown = maxBalanceSoFar > 0 
    ? ((maxBalanceSoFar - currentBalance) / maxBalanceSoFar) * 100 
    : 0;
  
  const isHighRisk = currentDrawdown > 15;

  const cardStyle = {
    background: '#1f2937',
    padding: '15px',
    borderRadius: '8px',
    border: '1px solid #374151',
    marginBottom: '10px'
  };

  const labelStyle = { fontSize: '11px', color: '#9ca3af', textTransform: 'uppercase', display:'flex', justifyContent:'space-between' };
  const valueStyle = { fontSize: '20px', fontWeight: 'bold', color: '#f3f4f6', margin: '5px 0' };

  return (
    <div>
      {/* Tarjeta Balance */}
      <div style={{...cardStyle, borderLeft: `4px solid ${isProfit ? '#10b981' : '#ef4444'}`}}>
        <div style={labelStyle}>
          <span>Balance Total</span> <DollarSign size={14}/>
        </div>
        <div style={valueStyle}>
            ${currentBalance.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </div>
        <div style={{fontSize: '12px', color: isProfit ? '#10b981' : '#ef4444'}}>
          {isProfit ? '+' : ''}{pnlPercentage.toFixed(2)}% ROI
        </div>
      </div>

      {/* Tarjeta Riesgo */}
      <div style={{...cardStyle, borderLeft: `4px solid ${isHighRisk ? '#ef4444' : '#6b7280'}`, marginBottom: 0}}>
        <div style={labelStyle}>
          <span>Max Drawdown</span> <AlertTriangle size={14}/>
        </div>
        <div style={valueStyle}>
            -{currentDrawdown.toFixed(2)}%
        </div>
        <div style={{fontSize: '12px', color: isHighRisk ? '#ef4444' : '#9ca3af'}}>
          {isHighRisk ? 'ALERTA: CRÍTICO' : 'Riesgo Controlado'}
        </div>
      </div>
    </div>
  );
};