/**
 * ML Performance Testing Script
 * Run in browser console to compare ML vs A* performance
 * Usage: await runMLPerformanceTests(numTests, layoutType)
 */

window.runMLPerformanceTests = async function(numTests = 20, layoutType = 'random') {
    console.log(`\n========== ML vs A* Performance Test (${numTests} runs) ==========\n`);
    
    if (!window.ui || !window.pathfindingManager) {
        console.error('UI or pathfinding manager not initialized');
        return;
    }

    const results = {
        ml: { nodes: [], times: [], distances: [], successes: 0 },
        astar: { nodes: [], times: [], distances: [], successes: 0 }
    };

    for (let i = 0; i < numTests; i++) {
        // Generate new layout
        window.ui.generateLayout(layoutType);
        
        // Wait a bit for layout to render
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Generate random start and end points
        let start, end;
        let attempts = 0;
        do {
            const r1 = Math.floor(Math.random() * window.ROWS);
            const c1 = Math.floor(Math.random() * window.COLS);
            const r2 = Math.floor(Math.random() * window.ROWS);
            const c2 = Math.floor(Math.random() * window.COLS);
            
            start = window.grid[r1][c1];
            end = window.grid[r2][c2];
            attempts++;
        } while ((start.isWall || end.isWall || (start.row === end.row && start.col === end.col)) && attempts < 10);

        if (start.isWall || end.isWall || (start.row === end.row && start.col === end.col)) {
            console.log(`Test ${i+1}/${numTests}: Skipped (couldn't find valid start/end)`);
            i--;
            continue;
        }

        window.start = start;
        window.end = end;

        console.log(`\nTest ${i+1}/${numTests}: Start=[${start.row},${start.col}] End=[${end.row},${end.col}]`);

        // Test A*
        try {
            const astarPf = new window.Pathfinder(
                { grid: window.grid, rows: window.ROWS, cols: window.COLS },
                window.AstarHeuristic
            );
            const astarStart = performance.now();
            const astarRes = await astarPf.findPath(start, end);
            const astarTime = performance.now() - astarStart;
            
            if (astarRes.success) {
                results.astar.successes++;
                results.astar.nodes.push(astarRes.nodesVisited);
                results.astar.times.push(astarTime);
                results.astar.distances.push(astarRes.distance);
                console.log(`  A*:       ${astarRes.nodesVisited} nodes, ${astarTime.toFixed(2)}ms, distance=${astarRes.distance.toFixed(2)}`);
            } else {
                console.log(`  A*:       FAILED`);
            }
        } catch (e) {
            console.error(`  A*:       ERROR: ${e.message}`);
        }

        // Test ML
        try {
            const mlPf = new window.Pathfinder(
                { grid: window.grid, rows: window.ROWS, cols: window.COLS },
                async (a, b) => await window.mlHeuristic(a, b, window.grid)
            );
            const mlStart = performance.now();
            const mlRes = await mlPf.findPath(start, end);
            const mlTime = performance.now() - mlStart;
            
            if (mlRes.success) {
                results.ml.successes++;
                results.ml.nodes.push(mlRes.nodesVisited);
                results.ml.times.push(mlTime);
                results.ml.distances.push(mlRes.distance);
                console.log(`  ML:       ${mlRes.nodesVisited} nodes, ${mlTime.toFixed(2)}ms, distance=${mlRes.distance.toFixed(2)}`);
            } else {
                console.log(`  ML:       FAILED`);
            }
        } catch (e) {
            console.error(`  ML:       ERROR: ${e.message}`);
        }

        // Compare this run
        if (results.astar.nodes.length > 0 && results.ml.nodes.length > 0) {
            const lastAstar = results.astar.nodes[results.astar.nodes.length - 1];
            const lastMl = results.ml.nodes[results.ml.nodes.length - 1];
            const ratio = (lastMl / lastAstar * 100).toFixed(1);
            const better = lastMl <= lastAstar ? '✓ ML better' : '✗ ML worse';
            console.log(`  ${better} (ML is ${ratio}% of A*)`);
        }
    }

    // Calculate statistics
    const calcStats = (arr) => {
        if (arr.length === 0) return { min: 0, max: 0, avg: 0, median: 0 };
        const sorted = [...arr].sort((a, b) => a - b);
        const sum = arr.reduce((a, b) => a + b, 0);
        const avg = sum / arr.length;
        const median = sorted[Math.floor(sorted.length / 2)];
        return { min: sorted[0], max: sorted[sorted.length - 1], avg, median };
    };

    console.log('\n========== SUMMARY ==========');
    console.log(`\nTests completed: ${numTests}`);
    console.log(`A* successes: ${results.astar.successes}/${numTests}`);
    console.log(`ML successes: ${results.ml.successes}/${numTests}`);

    if (results.astar.nodes.length > 0) {
        const astarStats = calcStats(results.astar.nodes);
        console.log(`\nA* Nodes Expanded:    min=${astarStats.min}, max=${astarStats.max}, avg=${astarStats.avg.toFixed(1)}, median=${astarStats.median}`);
        const astarTimeStats = calcStats(results.astar.times);
        console.log(`A* Runtime (ms):      min=${astarTimeStats.min.toFixed(2)}, max=${astarTimeStats.max.toFixed(2)}, avg=${astarTimeStats.avg.toFixed(2)}, median=${astarTimeStats.median.toFixed(2)}`);
    }

    if (results.ml.nodes.length > 0) {
        const mlStats = calcStats(results.ml.nodes);
        console.log(`\nML Nodes Expanded:    min=${mlStats.min}, max=${mlStats.max}, avg=${mlStats.avg.toFixed(1)}, median=${mlStats.median}`);
        const mlTimeStats = calcStats(results.ml.times);
        console.log(`ML Runtime (ms):      min=${mlTimeStats.min.toFixed(2)}, max=${mlTimeStats.max.toFixed(2)}, avg=${mlTimeStats.avg.toFixed(2)}, median=${mlTimeStats.median.toFixed(2)}`);
    }

    // Comparison
    if (results.astar.nodes.length > 0 && results.ml.nodes.length > 0) {
        const astarAvg = results.astar.nodes.reduce((a, b) => a + b, 0) / results.astar.nodes.length;
        const mlAvg = results.ml.nodes.reduce((a, b) => a + b, 0) / results.ml.nodes.length;
        const ratio = (mlAvg / astarAvg * 100).toFixed(1);
        
        console.log(`\n========== COMPARISON ==========`);
        console.log(`ML avg nodes / A* avg nodes: ${ratio}%`);
        
        if (mlAvg <= astarAvg) {
            console.log(`✓ ML performs equal or better than A* (${(100 - parseFloat(ratio)).toFixed(1)}% improvement)`);
        } else {
            console.log(`✗ ML performs worse than A* (${(parseFloat(ratio) - 100).toFixed(1)}% overhead)`);
        }

        const mlFaster = results.ml.times.reduce((a, b) => a + b, 0) / results.ml.times.length;
        const astarFaster = results.astar.times.reduce((a, b) => a + b, 0) / results.astar.times.length;
        const timeRatio = (mlFaster / astarFaster * 100).toFixed(1);
        console.log(`ML avg time / A* avg time: ${timeRatio}%`);
    }

    return results;
};

// Quick test: 10 runs on random layout
window.quickMLTest = async function() {
    console.log('Running quick ML performance test (10 runs)...');
    return await window.runMLPerformanceTests(10, 'random');
};

// Full test: 20 runs on multiple layouts
window.fullMLTest = async function() {
    console.log('Running full ML performance test...');
    const results = {};
    
    for (const layout of ['random', 'maze', 'clustered', 'mixed']) {
        console.log(`\n\n=============== Testing on ${layout.toUpperCase()} layout ===============`);
        results[layout] = await window.runMLPerformanceTests(20, layout);
        await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    return results;
};

console.log('ML Performance Test Script Loaded');
console.log('Usage:');
console.log('  window.quickMLTest()           - Quick test (10 runs)');
console.log('  window.fullMLTest()            - Full test (20 runs × 4 layouts)');
console.log('  window.runMLPerformanceTests(numTests, layoutType) - Custom test');
