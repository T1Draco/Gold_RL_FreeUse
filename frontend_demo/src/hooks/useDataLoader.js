import { useState, useEffect } from 'react';
import Papa from 'papaparse';

const INITIAL_BALANCE = 10000;

export const useDataLoader = () => {
    const [data, setData] = useState({ 
        marketData: [], 
        tradeEvents: {}, 
        totalSteps: 0, 
        initialStep: 0,
        firstTradeDate: null // Para saber cuándo termina el aprendizaje
    });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadFiles = async () => {
            try {
                // 1. CARGAR MERCADO COMPLETO (Sin recortar)
                const marketRes = await fetch('/data/XAUUSD_D1_rl.csv');
                const marketText = await marketRes.text();
                const marketParsed = Papa.parse(marketText, { header: true, skipEmptyLines: true }).data;

                const findKey = (obj, key) => Object.keys(obj).find(k => k.toLowerCase().includes(key.toLowerCase()));

                const fullMarketData = marketParsed.map(d => {
                    const dateKey = findKey(d, 'date') || findKey(d, 'time');
                    const closeKey = findKey(d, 'close');
                    if (!dateKey || !closeKey) return null;
                    return {
                        time: d[dateKey].split(' ')[0], 
                        value: parseFloat(d[closeKey])
                    };
                }).filter(d => d && !isNaN(d.value));

                // 2. CARGAR TRADES
                const tradesRes = await fetch('/data/trades_rppo.csv');
                const tradesText = await tradesRes.text();
                const tradesParsed = Papa.parse(tradesText, { header: true, skipEmptyLines: true }).data;

                const eventsMap = {};
                let firstTradeDateStr = null;
                let runningBalance = INITIAL_BALANCE;

                // Ordenar trades cronológicamente
                tradesParsed.sort((a, b) => {
                    const da = a[findKey(a, 'entry_date')];
                    const db = b[findKey(b, 'entry_date')];
                    return new Date(da) - new Date(db);
                });

                // Detectar fecha del primer trade real
                if (tradesParsed.length > 0) {
                    const firstT = tradesParsed[0];
                    firstTradeDateStr = firstT[findKey(firstT, 'entry_date')].split(' ')[0];
                }

                tradesParsed.forEach(t => {
                    const entryKey = findKey(t, 'entry_date');
                    const exitKey = findKey(t, 'exit_date');
                    const pnlKey = findKey(t, 'net_pnl') || findKey(t, 'pnl');

                    if (!t[entryKey] || !t[exitKey]) return;

                    const entryDate = t[entryKey].split(' ')[0];
                    const exitDate = t[exitKey].split(' ')[0];
                    const pnl = parseFloat(t[pnlKey]);

                    runningBalance += pnl;

                    if (!eventsMap[entryDate]) eventsMap[entryDate] = [];
                    eventsMap[entryDate].push({ ...t, type: 'ENTRY', original_type: t.type });

                    if (!eventsMap[exitDate]) eventsMap[exitDate] = [];
                    eventsMap[exitDate].push({ ...t, type: 'EXIT', pnl: pnl, balance: runningBalance });
                });

                console.log(`DATA: Cargado completo 1996-2025. Fase de aprendizaje hasta: ${firstTradeDateStr}`);

                setData({
                    marketData: fullMarketData, // Datos completos
                    tradeEvents: eventsMap,
                    totalSteps: fullMarketData.length - 1,
                    initialStep: 0, // Empezamos desde el día 1 (1996)
                    firstTradeDate: firstTradeDateStr
                });
                setLoading(false);

            } catch (e) {
                console.error("Error:", e);
                setLoading(false);
            }
        };

        loadFiles();
    }, []);

    return { ...data, loading, initialBalance: INITIAL_BALANCE };
};