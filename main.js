import { Cell } from './grid.js';
import { AstarHeuristic, Pathfinder, dijkastraHeuristic, greedyHeuristic, mlHeuristic, mlDynamicHeuristic } from './pathfinder.js';

// Configure TensorFlow.js to use CPU backend to avoid WebGL issues
async function initializeTensorFlow() {
    try {
        await window.tf.setBackend('cpu');
        console.log('TensorFlow.js backend set to CPU');
    } catch (error) {
        console.warn('Failed to set TensorFlow.js backend to CPU:', error);
    }
}

// Initialize TensorFlow.js when the page loads
initializeTensorFlow();

const canvas = document.getElementById('gridCanvas');
const ROWS = 20, COLS = 20;
const WIDTH = canvas.width, HEIGHT = canvas.height;
const cellSize = WIDTH / ROWS;
let grid = [];
let start = null, end = null;
let mouseDown = false, wallDrawMode = null;
let runHistory = [];
let setups = JSON.parse(localStorage.getItem('simpleSetups') || '{}');


function drawGrid(showCosts = false) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    for (let i = 0; i < ROWS; i++) {
        for (let j = 0; j < COLS; j++) {
            const cell = grid[i][j];
            ctx.fillStyle = cell.color;
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            ctx.strokeStyle = '#aaa';
            ctx.lineWidth = 1;
            ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);
            if (showCosts && !cell.isWall) {
                ctx.fillStyle = 'black';
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(cell.cost, j * cellSize + cellSize/2, i * cellSize + cellSize/2);
            }
        }
    }
}

function clearGrid(dynamic = false) { //clearing the grid to become blank
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
            row.push(cell);
        }
        grid.push(row);
    }
    start = null;
    end = null;
    runHistory = [];
    drawGrid(dynamic);
    updateTable();
}

//Pathfinding
function getSelectedAlgorithm() { //picking the algorithm based on the dropdown selection
    const algo = document.getElementById('algoSelect').value;
    if (algo === 'astar') return AstarHeuristic;
    if (algo === 'astar_dynamic') return AstarHeuristic; // use same heuristic, but grid has costs
    if (algo === 'dijkstra') return () => 0;
    if (algo === 'greedy') return (a, b) => Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
    if (algo === 'ml') return async (a, b) => await mlHeuristic(a, b, grid);
    if (algo === 'ml_dynamic') return async (a, b) => await mlDynamicHeuristic(a, b, grid);
    return AstarHeuristic; //default to A* if nothing is selected
}

async function runPathfinder() {
    if (!start || !end) {
        alert('Set start and end!');
        return;
    }
    const heuristic = getSelectedAlgorithm();
    const pf = new Pathfinder({ grid, rows: ROWS, cols: COLS }, heuristic);
    //log which algorithm is being run
    console.log('Selected algorithm:', document.getElementById('algoSelect').value);
    if (document.getElementById('algoSelect').value === 'ml') {
        console.log('Calling pf.runAsync (ML)...');
        await pf.runAsync(start, end, drawGrid, (success, nodesVisited, distanceTraveled) => {
            runHistory.push({ nodesVisited, distanceTraveled });
            updateTable();
            if (success) showPath();
        });
    } else {
        pf.run(start, end, drawGrid, (success, nodesVisited, distanceTraveled) => {
            runHistory.push({ nodesVisited, distanceTraveled });
            updateTable();
            if (success) showPath();
        });
    }
}

function drawGridAuto() {
    const algo = document.getElementById('algoSelect').value;
    const isDynamic = grid.some(row => row.some(cell => cell.cost !== 1));
    drawGrid((algo === 'astar_dynamic' || algo === 'ml_dynamic') && isDynamic);
}

function showPath() {
    let cell = end;
    while (cell && cell !== start) {
        if (cell !== end) cell.color = '#ff9800';
        cell = cell.parent;
    }
    drawGridAuto();
}


function updateTable() {
    const tbody = document.getElementById('resultsBody');
    tbody.innerHTML = '';
    runHistory.forEach((run, i) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${i + 1}</td><td>${run.nodesVisited}</td><td>${run.distanceTraveled}</td>`;
        tbody.appendChild(tr);
    });
}

//Setups of a bunch of stuff
function saveSetup() {
    const name = document.getElementById('setupName').value.trim();
    if (!name) {
        alert('Enter a setup name!');
        return;
    }
    setups[name] = {
        grid: grid.map(row => row.map(cell => ({ isWall: cell.isWall, cost: cell.cost }))), // save cell cost
        start: start ? { row: start.row, col: start.col } : null,
        end: end ? { row: end.row, col: end.col } : null
    };
    localStorage.setItem('simpleSetups', JSON.stringify(setups));
    updateSetupDropdown();
}

function loadSetup() {
    const select = document.getElementById('setupSelect');
    const name = select.value;
    if (!name || !setups[name]) return;
    const setup = setups[name];
    grid = [];
    for (let i = 0; i < ROWS; i++) {
        let row = [];
        for (let j = 0; j < COLS; j++) {
            let cell = new Cell(i, j);
            cell.isWall = setup.grid[i][j].isWall;
            cell.cost = setup.grid[i][j].cost !== undefined ? setup.grid[i][j].cost : 1;
            cell.weight = cell.cost;
            cell.color = cell.isWall ? 'black' : 'white';
            row.push(cell);
        }
        grid.push(row);
    }
    start = setup.start ? grid[setup.start.row][setup.start.col] : null;
    end = setup.end ? grid[setup.end.row][setup.end.col] : null;
    if (start) start.color = 'green';
    if (end) end.color = 'red';
    runHistory = [];
    drawGridAuto();
    updateTable();
}

function updateSetupDropdown() {
    const select = document.getElementById('setupSelect');
    select.innerHTML = '';
    Object.keys(setups).forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
    });
}

//option to export data to csv(excel)
function exportToExcel() {
    const table = document.querySelector('table');
    let csv = Array.from(table.rows).map(row => Array.from(row.cells).map(cell => cell.textContent).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pathfinding_results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

//mouse Events(Controls)
function handleMouse(e, isClick = false) {
    const rect = canvas.getBoundingClientRect();
    let clientX = e.touches ? e.touches[0].clientX : e.clientX;
    let clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;
    const row = Math.floor(y / cellSize);
    const col = Math.floor(x / cellSize);
    if (row < 0 || row >= ROWS || col < 0 || col >= COLS) return;
    const cell = grid[row][col];
    if (isClick) {
        if (!start && !cell.isWall) {
            start = cell; start.color = 'green';
        } else if (!end && cell !== start && !cell.isWall) {
            end = cell; end.color = 'red';
        } else if (cell !== start && cell !== end) {
            cell.isWall = !cell.isWall;
            cell.color = cell.isWall ? 'black' : 'white';
        }
    } else if (mouseDown && start && end && cell !== start && cell !== end) {
        if (wallDrawMode === 'add' && !cell.isWall) {
            cell.isWall = true; cell.color = 'black';
        } else if (wallDrawMode === 'remove' && cell.isWall) {
            cell.isWall = false; cell.color = 'white';
        }
    }
    drawGridAuto();
}

canvas.addEventListener('mousedown', e => {
    mouseDown = true;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    let clientX = e.touches ? e.touches[0].clientX : e.clientX;
    let clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;
    const row = Math.floor(y / cellSize);
    const col = Math.floor(x / cellSize);
    if (row >= 0 && row < ROWS && col >= 0 && col < COLS) {
        const cell = grid[row][col];
        if (start && end && cell !== start && cell !== end) {
            wallDrawMode = cell.isWall ? 'remove' : 'add';
        } else {
            wallDrawMode = null;
        }
    } else {
        wallDrawMode = null;
    }
    handleMouse(e, true);
});
canvas.addEventListener('mousemove', e => {
    if (mouseDown) handleMouse(e, false);
});
canvas.addEventListener('mouseup', () => {
    mouseDown = false;
    wallDrawMode = null;
});
canvas.addEventListener('mouseleave', () => {
    mouseDown = false;
    wallDrawMode = null;
});

window.addEventListener('DOMContentLoaded', () => {
    clearGrid();
    updateSetupDropdown();
    document.getElementById('clearBtn').onclick = () => clearGrid();
    document.getElementById('clearDynamicBtn').onclick = () => clearGrid(true); // new button for dynamic grid
    document.getElementById('startBtn').onclick = runPathfinder;
    document.getElementById('saveBtn').onclick = saveSetup;
    document.getElementById('loadBtn').onclick = loadSetup;
    document.getElementById('exportBtn').onclick = exportToExcel;
    // Add ML-Dynamic and A*-Dynamic to algorithm dropdown
    const algoSelect = document.getElementById('algoSelect');
    if (!Array.from(algoSelect.options).some(opt => opt.value === 'ml_dynamic')) {
        const opt = document.createElement('option');
        opt.value = 'ml_dynamic';
        opt.textContent = 'ML-Dynamic';
        algoSelect.appendChild(opt);
    }
    if (!Array.from(algoSelect.options).some(opt => opt.value === 'astar_dynamic')) {
        const opt = document.createElement('option');
        opt.value = 'astar_dynamic';
        opt.textContent = 'A*-Dynamic';
        algoSelect.appendChild(opt);
    }
    document.getElementById('algoSelect').onchange = function() {
        const algo = this.value;
        if (algo === 'astar_dynamic' || algo === 'ml_dynamic') {
            // Show costs if grid is dynamic
            const isDynamic = grid.some(row => row.some(cell => cell.cost !== 1));
            drawGrid(isDynamic);
            if (!isDynamic) {
                alert('Grid is not dynamic! Press "Clear Dynamic Grid" to generate a dynamic environment.');
            }
        } else if (algo === 'astar' || algo === 'ml' || algo === 'dijkstra' || algo === 'greedy') {
            // Hide costs, but keep grid as-is
            drawGrid(false);
        }
    };
});
