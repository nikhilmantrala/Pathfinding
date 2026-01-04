import { Cell } from './grid.js';
import { AstarHeuristic, Pathfinder, dijkastraHeuristic, greedyHeuristic } from './pathfinder.js';
import { mlHeuristic, mlDynamicHeuristic } from './ml-heuristic.js';
import { GridGenerator } from './gridGenerator.js';
import { BatchTestManager } from './batch-test-manager.js';
import { PathfindingManager } from './pathfinding-manager.js';
import { UIManager } from './ui-manager.js';
import { BatchTestConfig } from './batch-test-config.js';

// Constants
const ROWS = 20, COLS = 20;
const THROTTLE_TIME = 50; // ms between pathfinder calls

// Initialize variables
let canvas = null;
let cellSize = 0;
let WIDTH = 0;
let HEIGHT = 0;
let grid = [];
let start = null;
let end = null;
let mouseDown = false;
let wallDrawMode = null;
let setups = {};
let gridGenerator = null;
let pathfindingManager = null;
let uiManager = null;
let batchTestManager = null;
let lastPathfinderCall = 0;

// Wait for DOM to be loaded
document.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('gridCanvas');
    if (!canvas) {
        console.error('Canvas element not found');
        return;
    }

    WIDTH = canvas.width;
    HEIGHT = canvas.height;
    cellSize = WIDTH / ROWS;

    // Initialize grid
    grid = Array(ROWS).fill().map((_, row) => 
        Array(COLS).fill().map((_, col) => new Cell(row, col))
    );

    // Initialize managers
    gridGenerator = new GridGenerator(grid, ROWS, COLS);
    pathfindingManager = new PathfindingManager(grid, ROWS, COLS);
    uiManager = new UIManager(canvas, ROWS, COLS);
    
    // Initialize batch test manager and export functionality
    batchTestManager = new BatchTestManager(pathfindingManager, uiManager, gridGenerator);
    const updateExportButton = uiManager.setupExportButton(batchTestManager);
    
    // Add listener to update export button state after tests complete
    const batchTestBtn = document.getElementById('batchTestBtn');
    if (batchTestBtn) {
        batchTestBtn.addEventListener('click', async () => {
            try {
                // Hide obstacle test results when starting bulk tests
                const obstacleTestResults = document.getElementById('obstacleTestResults');
                const testSummary = document.getElementById('testSummary');
                if (obstacleTestResults) obstacleTestResults.style.display = 'none';
                if (testSummary) testSummary.style.display = 'none';
                
                await batchTestManager.runTests(BatchTestConfig);
                if (updateExportButton) updateExportButton();
            } catch (error) {
                console.error('Error running batch tests:', error);
            }
        });
    }
    
    // Add listener for A* vs ML Obstacle comparison button
    const obstacleTestBtn = document.getElementById('obstacleTestBtn');
    if (obstacleTestBtn) {
        obstacleTestBtn.addEventListener('click', async () => {
            try {
                // Hide bulk test results and summary when starting obstacle test
                const resultsDiv = document.getElementById('results');
                const testSummary = document.getElementById('testSummary');
                if (resultsDiv) resultsDiv.style.display = 'none';
                if (testSummary) testSummary.style.display = 'none';
                
                await batchTestManager.runObstacleComparison();
            } catch (error) {
                console.error('Error running obstacle comparison:', error);
            }
        });
    }

    // Load saved setups from localStorage
    setups = JSON.parse(localStorage.getItem('simpleSetups') || '{}');

    // Add mouse event listeners
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseUp);

    // Initialize components
    initGrid();
    
    // Initialize batch test controls after ensuring UI manager is ready
    if (uiManager && uiManager.elements && uiManager.elements.controls && uiManager.elements.controls.batchTestBtn) {
        initializeBatchTestControls();
    } else {
        console.error('UI elements not properly initialized');
    }

    // Draw initial grid
    drawGrid();
});

async function initializeTensorFlow() {
    try {
        await window.tf.setBackend('webgl');  // Use WebGL for better performance
    } catch (error) {
        // Silently fall back to default backend
    }
}

initializeTensorFlow();

// Mouse interaction handlers
function getCellFromMouseEvent(e) {
    if (!canvas || !grid) return null;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);
    if (row >= 0 && row < ROWS && col >= 0 && col < COLS && grid[row] && grid[row][col]) {
        return grid[row][col];
    }
    return null;
}

function handleMouseDown(e) {
    const cell = getCellFromMouseEvent(e);
    if (!cell) return;

    mouseDown = true;
    if (!start && !cell.isWall && !cell.isEnd) {
        start = cell;
        cell.isStart = true;
        wallDrawMode = null;
    } else if (!end && !cell.isWall && !cell.isStart) {
        end = cell;
        cell.isEnd = true;
        wallDrawMode = null;
    } else if (!cell.isStart && !cell.isEnd) {
        wallDrawMode = !cell.isWall;
        cell.isWall = wallDrawMode;
    }
    drawGrid();
}

function handleMouseMove(e) {
    if (!mouseDown || wallDrawMode === null) return;
    const cell = getCellFromMouseEvent(e);
    if (!cell || cell.isStart || cell.isEnd) return;
    
    cell.isWall = wallDrawMode;
    drawGrid();
}

function handleMouseUp() {
    mouseDown = false;
}

function drawGrid(showCosts = false) {
    if (!canvas) return;  // Don't try to draw if canvas isn't ready
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;  // Don't try to draw if context can't be obtained
    
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    
    // Draw in batches for better performance
    const wallCells = [];
    const costCells = [];
    const normalCells = [];
    
    for (let i = 0; i < ROWS; i++) {
        for (let j = 0; j < COLS; j++) {
            const cell = grid[i][j];
            if (cell.isWall) {
                wallCells.push({ cell, x: j * cellSize, y: i * cellSize });
            } else if (cell.cost > 1) {
                costCells.push({ cell, x: j * cellSize, y: i * cellSize });
            } else {
                normalCells.push({ cell, x: j * cellSize, y: i * cellSize });
            }
        }
    }
    
    // Draw cells in batches
    const drawBatch = (cells) => {
        cells.forEach(({ cell, x, y }) => {
            // Set fill style based on cell state
            if (cell.isStart) {
                ctx.fillStyle = '#00ff00';  // Green for start
            } else if (cell.isEnd) {
                ctx.fillStyle = '#ff0000';  // Red for end
            } else if (cell.isWall) {
                ctx.fillStyle = '#000000';  // Black for walls
            } else if (cell.isPath) {
                ctx.fillStyle = '#0000ff';  // Blue for path
            } else if (cell.isVisited) {
                ctx.fillStyle = '#aaaaff';  // Light blue for visited
            } else {
                ctx.fillStyle = '#ffffff';  // White for empty cells
            }
            
            ctx.fillRect(x, y, cellSize, cellSize);
            ctx.strokeStyle = '#aaa';
            ctx.strokeRect(x, y, cellSize, cellSize);
            if (showCosts && !cell.isWall) {
                ctx.fillStyle = 'black';
                ctx.fillText(cell.cost.toFixed(1), x + cellSize/2, y + cellSize/2);
            }
        });
    };
    
    // Set text properties once
    if (showCosts) {
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
    }
    
    // Draw in order: normal, cost, wall cells
    drawBatch(normalCells);
    drawBatch(costCells);
    drawBatch(wallCells);
}

// Moved to BatchTestManager

// Other existing functions remain the same...

document.getElementById('startBtn').addEventListener('click', runPathfinder);
document.getElementById('clearBtn').addEventListener('click', () => clearGrid(false));
document.getElementById('clearDynamicBtn').addEventListener('click', () => clearGrid(true));
document.getElementById('generateBtn').addEventListener('click', () => {
    if (!gridGenerator) {
        gridGenerator = new GridGenerator({ grid, rows: ROWS, cols: COLS });
    }
    const gridType = document.getElementById('gridTypeSelect').value;
    gridGenerator.generate(gridType);
    start = null;
    end = null;
    drawGrid(true);
});

async function runTestWithAlgorithm(batchSize, algo, gridLayouts, elements, statsTable) {
    const stats = {
        totalNodes: 0,
        totalDistance: 0,
        totalTime: 0,
        runs: 0,
        successful: 0,
        minNodes: Infinity,
        maxNodes: 0,
        minDist: Infinity,
        maxDist: 0
    };

    for (let i = 0; i < gridLayouts.length; i++) {
        const { validPositions, startPos, endPos } = gridLayouts[i];
        const startTime = performance.now();

        // Set up the grid from saved layout
        grid = gridLayouts[i].grid;
        start = grid[startPos.y][startPos.x];
        end = grid[endPos.y][endPos.x];

        try {
            const pathfinder = new Pathfinder({ grid, rows: ROWS, cols: COLS }, getSelectedAlgorithm(algo));
            const result = await new Promise((resolve) => {
                const callback = (success, nodes, dist) => {
                    resolve({
                        success,
                        nodes,
                        distance: parseFloat(dist || 0),
                        time: performance.now() - startTime
                    });
                };

                if (algo === 'ml' || algo === 'ml_variable_cost') {
                    pathfinder.runAsync(start, end, null, callback);
                } else {
                    pathfinder.run(start, end, null, callback);
                }
            });

            if (result.success) {
                stats.successful++;
                stats.totalNodes += result.nodes;
                stats.totalDistance += result.distance;
                stats.totalTime += result.time;
                stats.minNodes = Math.min(stats.minNodes, result.nodes);
                stats.maxNodes = Math.max(stats.maxNodes, result.nodes);
                stats.minDist = Math.min(stats.minDist, result.distance);
                stats.maxDist = Math.max(stats.maxDist, result.distance);
            }
            stats.runs++;

            const row = statsTable.insertRow();
            row.insertCell().textContent = `${algo} #${i + 1}`;
            row.insertCell().textContent = result.time.toFixed(2);
            row.insertCell().textContent = result.nodes;
            row.insertCell().textContent = result.success ? result.distance.toFixed(2) : 'N/A';
            row.insertCell().textContent = result.success ? '✓' : '✗';
        } catch (error) {
            console.error(`Error in test run for ${algo}:`, error);
            const row = statsTable.insertRow();
            row.insertCell().textContent = `${algo} #${i + 1}`;
            row.insertCell().textContent = 'Error';
            row.insertCell().textContent = 'Error';
            row.insertCell().textContent = 'Error';
            row.insertCell().textContent = '✗';
        }

        if (i % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0));
    }

    return stats;
}

// Function to reset grid to initial state
function resetGrid() {
    grid = Array(ROWS).fill().map((_, row) => 
        Array(COLS).fill().map((_, col) => new Cell(row, col))
    );
    start = null;
    end = null;
    gridGenerator = new GridGenerator(grid, ROWS, COLS);
    drawGrid();
}

function initializeBatchTestControls() {
    if (!uiManager || !uiManager.elements || !uiManager.elements.batchTest) {
        console.error('UI Manager not properly initialized');
        return;
    }

    batchTestManager = new BatchTestManager(pathfindingManager, uiManager, gridGenerator);
    
    // Set up export button
    const exportBtn = uiManager.elements.batchTest.exportBtn;
    if (exportBtn) {
        exportBtn.addEventListener('click', () => batchTestManager.exportToExcel());
        exportBtn.disabled = true;
    }
    
    // Initialize algorithm pairs dropdown
    const algorithmPairSelect = uiManager.elements.batchTest.algorithmPairSelect;
    if (algorithmPairSelect) {
        // Clear existing options
        algorithmPairSelect.innerHTML = '';
        
        // Add algorithm pairs to dropdown
        BatchTestConfig.ALGORITHM_PAIRS.forEach((pair, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = pair.name;
            algorithmPairSelect.appendChild(option);
        });
    }

    // Add event listener for batch test button
    // Initialize layout families checkboxes
    const layoutFamilies = uiManager.elements.batchTest.layoutFamilies;
    if (layoutFamilies) {
        // Clear existing checkboxes
        layoutFamilies.innerHTML = '';
        
        // Add layout family checkboxes
        Object.entries(BatchTestConfig.LAYOUT_FAMILIES).forEach(([key, value]) => {
            const div = document.createElement('div');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `layout-${key}`;
            checkbox.value = value;
            checkbox.checked = true;  // Default to checked
            
            const label = document.createElement('label');
            label.htmlFor = `layout-${key}`;
            label.textContent = value;
            
            div.appendChild(checkbox);
            div.appendChild(label);
            layoutFamilies.appendChild(div);
        });
    }

    // Initialize density bands checkboxes
    const densityBands = uiManager.elements.batchTest.densityBands;
    if (densityBands) {
        // Clear existing checkboxes
        densityBands.innerHTML = '';
        
        // Add density band checkboxes
        Object.entries(BatchTestConfig.DENSITY_BANDS).forEach(([key, value]) => {
            const div = document.createElement('div');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `density-${key}`;
            checkbox.value = value;
            checkbox.checked = true;  // Default to checked
            
            const label = document.createElement('label');
            label.htmlFor = `density-${key}`;
            label.textContent = value;
            
            div.appendChild(checkbox);
            div.appendChild(label);
            densityBands.appendChild(div);
        });
    }

    // Add event listener for batch test button
    if (uiManager.elements.controls.batchTestBtn) {
        uiManager.elements.controls.batchTestBtn.addEventListener('click', async () => {
            try {
                // Get selected layout families
                const selectedLayouts = Array.from(uiManager.elements.batchTest.layoutFamilies.querySelectorAll('input[type="checkbox"]:checked'))
                    .map(cb => cb.value);
                
                // Get selected density bands
                const selectedDensities = Array.from(uiManager.elements.batchTest.densityBands.querySelectorAll('input[type="checkbox"]:checked'))
                    .map(cb => cb.value);
                
                if (selectedLayouts.length === 0) {
                    throw new Error('Please select at least one layout family');
                }
                if (selectedDensities.length === 0) {
                    throw new Error('Please select at least one density band');
                }
                
                // Get seedsPerFamily based on preset selection
                const seedPresetValue = uiManager.elements.batchTest.seedPreset?.value || 'RECOMMENDED';
                let seedsPerFamily;
                switch (seedPresetValue) {
                    case 'QUICK':
                        seedsPerFamily = 5;
                        break;
                    case 'ROBUST':
                        seedsPerFamily = 50;
                        break;
                    case 'RECOMMENDED':
                    default:
                        seedsPerFamily = 20;
                        break;
                }
                
                const config = {
                    seedsPerFamily: seedsPerFamily,
                    pairsPerSeed: parseInt(uiManager.elements.batchTest.pairsPerSeed?.value) || 3,
                    timeoutMs: parseInt(uiManager.elements.batchTest.timeoutMs?.value) || 10000,
                    saveInputs: uiManager.elements.batchTest.saveInputs?.checked ?? true,
                    algorithmPair: BatchTestConfig.ALGORITHM_PAIRS[parseInt(uiManager.elements.batchTest.algorithmPairSelect?.value) || 0],
                    layouts: selectedLayouts,
                    densities: selectedDensities
                };
                
                if (!config.algorithmPair) {
                    throw new Error('No algorithm pair selected');
                }
                
                await batchTestManager.runTests(config);
            } catch (error) {
                console.error('Error running batch tests:', error);
                alert('An error occurred while running batch tests. Check the console for details.');
            }
        });
    } else {
        console.error('Batch test button not found');
    }
    
    // Setup save/load functionality
    setupSaveLoadHandlers();
}

// Save/Load Setup Functions
function setupSaveLoadHandlers() {
    const saveBtn = document.getElementById('saveBtn');
    const loadBtn = document.getElementById('loadBtn');
    
    if (saveBtn) {
        saveBtn.addEventListener('click', saveSetup);
    }
    
    if (loadBtn) {
        loadBtn.addEventListener('click', loadSetup);
    }
    
    // Update dropdown on load
    updateSetupDropdown();
}

function saveSetup() {
    const setupName = document.getElementById('setupName');
    const name = setupName?.value.trim();
    
    if (!name) {
        alert('Please enter a setup name!');
        return;
    }
    
    // Save current grid state
    setups[name] = {
        grid: grid.map(row => row.map(cell => ({ 
            isWall: cell.isWall, 
            cost: cell.cost 
        }))),
        start: start ? { row: start.row, col: start.col } : null,
        end: end ? { row: end.row, col: end.col } : null
    };
    
    // Save to localStorage
    localStorage.setItem('simpleSetups', JSON.stringify(setups));
    
    // Update dropdown and clear input
    updateSetupDropdown();
    setupName.value = '';
    
    alert(`Setup "${name}" saved successfully!`);
}

function loadSetup() {
    const setupSelect = document.getElementById('setupSelect');
    const name = setupSelect?.value;
    
    if (!name || !setups[name]) {
        alert('Please select a setup to load!');
        return;
    }
    
    const setup = setups[name];
    
    // Restore grid from saved state
    for (let i = 0; i < ROWS; i++) {
        for (let j = 0; j < COLS; j++) {
            grid[i][j].isWall = setup.grid[i][j].isWall;
            grid[i][j].cost = setup.grid[i][j].cost !== undefined ? setup.grid[i][j].cost : 1;
            grid[i][j].weight = grid[i][j].cost;
            grid[i][j].color = grid[i][j].isWall ? 'black' : 'white';
        }
    }
    
    // Restore start and end points
    start = setup.start ? grid[setup.start.row][setup.start.col] : null;
    end = setup.end ? grid[setup.end.row][setup.end.col] : null;
    
    if (start) start.color = 'green';
    if (end) end.color = 'red';
    
    // Update pathfinding manager
    pathfindingManager.start = start;
    pathfindingManager.end = end;
    pathfindingManager.runHistory = [];
    
    // Redraw grid
    drawGrid();
    updateTable();
    
    alert(`Setup "${name}" loaded successfully!`);
}

function updateSetupDropdown() {
    const setupSelect = document.getElementById('setupSelect');
    if (!setupSelect) return;
    
    setupSelect.innerHTML = '';
    
    const setupNames = Object.keys(setups);
    if (setupNames.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No saved setups';
        setupSelect.appendChild(option);
        return;
    }
    
    setupNames.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        setupSelect.appendChild(option);
    });
}

// All batch test functionality has been moved to BatchTestManager

// Function to get selected algorithm
function getSelectedAlgorithm(algorithm = null) {
    if (algorithm) {
        switch (algorithm) {
            case 'astar': return AstarHeuristic;
            case 'astar_variable_cost': return AstarHeuristic;
            case 'dijkstra': return () => 0;  // Dijkstra uses 0 as heuristic
            case 'greedy': return (a, b) => Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
            case 'ml': return async (a, b) => await mlHeuristic(a, b, grid);
            case 'ml_variable_cost': return async (a, b) => await mlDynamicHeuristic(a, b, grid);
            default: return AstarHeuristic;
        }
    }
    const algoSelect = document.getElementById('algoSelect');
    return getSelectedAlgorithm(algoSelect.value);
}

async function runPathfinder() {
    if (!start || !end) {
        alert('Set start and end points!');
        return;
    }

    const now = performance.now();
    if (now - lastPathfinderCall < THROTTLE_TIME) {
        return; // Throttle calls
    }
    lastPathfinderCall = now;

    const heuristic = getSelectedAlgorithm();
    const pf = new Pathfinder({ grid, rows: ROWS, cols: COLS }, heuristic);
    const algo = document.getElementById('algoSelect').value;
    const startTime = performance.now();

    const recordResult = (success, nodesVisited, distanceTraveled) => {
        const timeMs = performance.now() - startTime;
        // Attempt to read layout type and wall density from UI, fallback to N/A
        const layoutType = document.getElementById('gridTypeSelect')?.value || 'N/A';
        const wallDensityRaw = document.getElementById('wallDensity')?.value; // optional input
        const wallDensity = wallDensityRaw !== undefined ? (isNaN(parseFloat(wallDensityRaw)) ? wallDensityRaw : parseFloat(wallDensityRaw)) : 'N/A';

        const entry = {
            algorithm: algo,
            layoutType,
            wallDensity,
            nodesVisited: typeof nodesVisited === 'number' ? nodesVisited : (nodesVisited || null),
            distanceTraveled: success ? (distanceTraveled ? parseFloat(distanceTraveled) : 0) : null,
            timeMs,
            success: !!success
        };
        // Use manager API which accepts objects
        pathfindingManager.addToHistory(entry);
        // Update UI via UIManager
        if (uiManager && typeof uiManager.updateResults === 'function') {
            uiManager.updateResults(pathfindingManager.getRunHistory());
        } else {
            updateTable();
        }
        if (success) showPath();
    };

    if (algo === 'ml' || algo === 'ml_variable_cost') {
        await pf.runAsync(start, end, drawGridAuto, (success, nodesVisited, distanceTraveled) => {
            recordResult(success, nodesVisited, distanceTraveled);
        });
    } else {
        pf.run(start, end, drawGridAuto, (success, nodesVisited, distanceTraveled) => {
            recordResult(success, nodesVisited, distanceTraveled);
        });
    }
}

function updateTable() {
    if (uiManager && typeof uiManager.updateResults === 'function') {
        uiManager.updateResults(pathfindingManager.getRunHistory());
        return;
    }

    const tbody = document.getElementById('resultsBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    pathfindingManager.getRunHistory().forEach((run, i) => {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = i + 1;
        row.insertCell(1).textContent = run.algorithm || 'unknown';
        row.insertCell(2).textContent = run.nodesVisited ?? 'N/A';
        row.insertCell(3).textContent = typeof run.distanceTraveled === 'number' ? run.distanceTraveled.toFixed(2) : 'No path';
        row.insertCell(4).textContent = typeof run.timeMs === 'number' ? run.timeMs.toFixed(2) : 'N/A';
    });
}

function drawGridAuto(visitedGrid) {
    if (visitedGrid) {
        // Update visited cells from the pathfinder's grid
        for (let i = 0; i < ROWS; i++) {
            for (let j = 0; j < COLS; j++) {
                if (!grid[i][j].isStart && !grid[i][j].isEnd && !grid[i][j].isWall) {
                    grid[i][j].isVisited = visitedGrid[i][j].isVisited;
                }
            }
        }
    }
    drawGrid(document.getElementById('algoSelect').value.includes('dynamic'));
}

function showPath() {
    // Reset all path and visited flags
    for (let i = 0; i < ROWS; i++) {
        for (let j = 0; j < COLS; j++) {
            const cell = grid[i][j];
            cell.isPath = false;
            if (!cell.isStart && !cell.isEnd && !cell.isWall) {
                cell.isVisited = false;
            }
        }
    }

    // Mark the path
    let current = end;
    while (current && current !== start) {
        current.isPath = true;
        current = current.parent;
    }

    drawGridAuto();
}

function clearGrid(dynamic = false) {
    grid = [];
    for (let i = 0; i < ROWS; i++) {
        let row = [];
        for (let j = 0; j < COLS; j++) {
            let cell = new Cell(i, j);
            if (dynamic) {
                // 10% chance to be slow terrain (cost 3), 20% chance cost 2, else cost 1
                const rand = Math.random();
                if (rand < 0.1) {
                    cell.cost = 3;
                } else if (rand < 0.3) {
                    cell.cost = 2;
                } else {
                    cell.cost = 1;
                }
            } else {
                cell.cost = 1;
            }
            cell.weight = cell.cost; // ensure weight is set for pathfinder
            cell.isPath = false;
            cell.isVisited = false;
            row.push(cell);
        }
        grid.push(row);
    }
    start = null;
    end = null;
    pathfindingManager.runHistory = [];
    pathfindingManager.start = null;
    pathfindingManager.end = null;
    gridGenerator = new GridGenerator(grid, ROWS, COLS);
    drawGrid(dynamic);
    updateTable();
}

function initGrid() {
    if (!canvas) return;  // Don't initialize if canvas isn't ready
    
    grid = [];
    for (let i = 0; i < ROWS; i++) {
        grid[i] = [];
        for (let j = 0; j < COLS; j++) {
            grid[i][j] = new Cell(i, j);
        }
    }
    gridGenerator = new GridGenerator({ grid, rows: ROWS, cols: COLS });
    start = null;
    end = null;
    mouseDown = false;
    wallDrawMode = null;
    drawGrid();
}

// Utility: run a quick comparison between A* and ML heuristics on the current grid/start/end
// Usage (in browser console): window.compareAlgorithms()
window.compareAlgorithms = async function() {
    if (!start || !end) {
        return null;
    }

    const results = [];
    const algos = [
        { name: 'astar', fn: AstarHeuristic },
        { name: 'ml', fn: async (a,b) => await mlHeuristic(a,b,grid) }
    ];

    for (const alg of algos) {
        const pf = new Pathfinder({ grid, rows: ROWS, cols: COLS }, alg.fn);
        const startTime = performance.now();
        const res = await pf.findPath(start, end);
        const timeMs = performance.now() - startTime;
        results.push({ algorithm: alg.name, success: res.success, nodes: res.nodesVisited, distance: res.distance, timeMs });
    }

    console.table(results);
    return results;
};
