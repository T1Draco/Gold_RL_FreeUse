import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, AreaSeries } from 'lightweight-charts';

export const TradingChart = ({ data, currentStep, tradeMarkers }) => {
    const containerRef = useRef();
    const chartInstance = useRef(null);
    const seriesInstance = useRef(null);

    // 1. CREACIÓN DEL GRÁFICO
    useEffect(() => {
        if (!containerRef.current || !data) return;

        if (chartInstance.current) {
            chartInstance.current.remove();
        }

        const chart = createChart(containerRef.current, {
            layout: { background: { type: ColorType.Solid, color: '#0b0f19' }, textColor: '#6b7280' },
            grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight,
            timeScale: { borderColor: '#1f2937', timeVisible: true },
            rightPriceScale: { borderColor: '#1f2937' },
            handleScale: { mouseWheel: false, pinch: false }, // Bloqueamos zoom manual para que no se rompa la vista
        });

        const series = chart.addSeries(AreaSeries, {
            lineColor: '#3b82f6',
            topColor: 'rgba(59, 130, 246, 0.4)',
            bottomColor: 'rgba(59, 130, 246, 0.0)',
            lineWidth: 2,
        });

        series.setData(data);
        chartInstance.current = chart;
        seriesInstance.current = series;

        // --- FORZAR ZOOM INICIAL INMEDIATO ---
        // Esto evita que se vea todo el historial comprimido al principio
        if (data.length > 0 && currentStep > 0) {
            const range = 80;
            const start = Math.max(0, currentStep - range + 20);
            const end = Math.min(data.length - 1, currentStep + 10);
            chart.timeScale().setVisibleRange({ from: data[start].time, to: data[end].time });
        }

        const resizeObserver = new ResizeObserver(entries => {
            if (!chartInstance.current) return;
            for (let entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    chartInstance.current.applyOptions({ width, height });
                    // Re-aplicar zoom al redimensionar
                    const timeScale = chartInstance.current.timeScale();
                    if (data[currentStep]) {
                         const start = Math.max(0, currentStep - 80 + 20);
                         const end = Math.min(data.length - 1, currentStep + 10);
                         timeScale.setVisibleRange({ from: data[start].time, to: data[end].time });
                    }
                }
            }
        });

        resizeObserver.observe(containerRef.current);

        return () => {
            resizeObserver.disconnect();
            if (chartInstance.current) {
                chartInstance.current.remove();
                chartInstance.current = null;
            }
        };
    }, [data]); // Solo al cargar datos

    // 2. ACTUALIZACIÓN (Al mover el slider o dar Play)
    useEffect(() => {
        if (!chartInstance.current || !seriesInstance.current || !data[currentStep]) return;

        try {
            // Zoom Dinámico que sigue al precio
            const timeScale = chartInstance.current.timeScale();
            const range = 80; 
            const start = Math.max(0, currentStep - range + 20);
            const end = Math.min(data.length - 1, currentStep + 10);
            
            timeScale.setVisibleRange({ from: data[start].time, to: data[end].time });

            // Marcadores (Flechas)
            const dateNow = new Date(data[currentStep].time);
            const visibleMarkers = tradeMarkers
                .filter(m => new Date(m.time) <= dateNow)
                .map(m => ({
                    time: m.time,
                    position: m.type === 'ENTRY' ? 'belowBar' : 'aboveBar',
                    color: m.type === 'ENTRY' ? '#eab308' : (m.pnl > 0 ? '#10b981' : '#ef4444'),
                    shape: m.type === 'ENTRY' ? 'arrowUp' : 'arrowDown',
                    text: m.type === 'ENTRY' ? 'BUY' : (m.pnl > 0 ? 'WIN' : 'LOSS'),
                    size: 2
                }));

            if (seriesInstance.current.setMarkers) {
                seriesInstance.current.setMarkers(visibleMarkers);
            }
        } catch(e) {}

    }, [currentStep, tradeMarkers, data]);

    return <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }} />;
};