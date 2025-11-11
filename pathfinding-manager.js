import { Cell } from './grid.js';
import { AstarHeuristic, dijkastraHeuristic, greedyHeuristic } from './pathfinder.js';
import { mlHeuristic, mlDynamicHeuristic } from './ml-heuristic.js';

export class PathfindingManager {
    constructor(grid, ROWS, COLS) {
        this.grid = grid;
        this.ROWS = ROWS;
        this.COLS = COLS;
        this.start = null;
        this.end = null;
        this.runHistory = [];
    }

    getSelectedAlgorithm(algoName) {
        switch (algoName) {
            case 'astar': return AstarHeuristic;
            case 'astar_dynamic': return AstarHeuristic;
            case 'dijkstra': return dijkastraHeuristic;
            case 'greedy': return greedyHeuristic;
            case 'ml': return async (a, b) => await mlHeuristic(a, b, this.grid);
            case 'ml_dynamic': return async (a, b) => await mlDynamicHeuristic(a, b, this.grid);
            default: return AstarHeuristic;
        }
    }

    setStartPoint(cell) {
        if (!cell.isWall && this.end !== cell) {
            this.start = cell;
            cell.color = 'green';
            return true;
        }
        return false;
    }

    setEndPoint(cell) {
        if (!cell.isWall && this.start !== cell) {
            this.end = cell;
            cell.color = 'red';
            return true;
        }
        return false;
    }

    clearGrid(dynamic = false) {
        this.grid = [];
        for (let i = 0; i < this.ROWS; i++) {
            let row = [];
            for (let j = 0; j < this.COLS; j++) {
                let cell = new Cell(i, j);
                if (dynamic) {
                    const rand = Math.random();
                    cell.cost = rand < 0.1 ? 3 : rand < 0.3 ? 2 : 1;
                } else {
                    cell.cost = 1;
                }
                cell.weight = cell.cost;
                row.push(cell);
            }
            this.grid.push(row);
        }
        this.start = null;
        this.end = null;
        this.runHistory = [];
        return this.grid;
    }

    showPath() {
        if (!this.end || !this.start) return;
        
        let cell = this.end;
        while (cell && cell !== this.start) {
            if (cell !== this.end) {
                cell.color = '#ff9800';
            }
            cell = cell.parent;
        }
    }

    validateSetup(setup) {
        if (!setup || !setup.grid || !Array.isArray(setup.grid)) return false;
        if (setup.grid.length !== this.ROWS) return false;
        if (setup.grid[0].length !== this.COLS) return false;
        return true;
    }

    loadSetup(setup) {
        if (!this.validateSetup(setup)) return false;

        this.grid = [];
        for (let i = 0; i < this.ROWS; i++) {
            let row = [];
            for (let j = 0; j < this.COLS; j++) {
                let cell = new Cell(i, j);
                cell.isWall = setup.grid[i][j].isWall;
                cell.cost = setup.grid[i][j].cost ?? 1;
                cell.weight = cell.cost;
                cell.color = cell.isWall ? 'black' : 'white';
                row.push(cell);
            }
            this.grid.push(row);
        }

        if (setup.start) {
            this.start = this.grid[setup.start.row][setup.start.col];
            this.start.color = 'green';
        } else {
            this.start = null;
        }

        if (setup.end) {
            this.end = this.grid[setup.end.row][setup.end.col];
            this.end.color = 'red';
        } else {
            this.end = null;
        }

        this.runHistory = [];
        return true;
    }

    saveSetup() {
        return {
            grid: this.grid.map(row => 
                row.map(cell => ({ 
                    isWall: cell.isWall, 
                    cost: cell.cost 
                }))
            ),
            start: this.start ? { row: this.start.row, col: this.start.col } : null,
            end: this.end ? { row: this.end.row, col: this.end.col } : null
        };
    }

    addToHistory(nodesVisitedOrObj, distanceTraveled) {
        // Accept either (nodesVisited, distanceTraveled) or an object with full info
        if (typeof nodesVisitedOrObj === 'object') {
            this.runHistory.push(nodesVisitedOrObj);
        } else {
            this.runHistory.push({ nodesVisited: nodesVisitedOrObj, distanceTraveled });
        }
    }

    getRunHistory() {
        return this.runHistory;
    }
}
