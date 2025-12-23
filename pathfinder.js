// Heuristic functions
export function manhattanDistance(a, b) {
    return Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
}

export function euclideanDistance(a, b) {
    return Math.sqrt(Math.pow(a.row - b.row, 2) + Math.pow(a.col - b.col, 2));
}

export function diagonalDistance(a, b) {
    const dx = Math.abs(a.row - b.row);
    const dy = Math.abs(a.col - b.col);
    return Math.max(dx, dy);
}

// Octile distance - optimal heuristic for 8-directional movement with unit costs
export function octileDistance(a, b) {
    const dx = Math.abs(a.row - b.row);
    const dy = Math.abs(a.col - b.col);
    const D = 1;
    const D2 = Math.SQRT2;
    return D * (dx + dy) + (D2 - 2 * D) * Math.min(dx, dy);
}

// Algorithm-specific heuristic functions
// FIXED: Use octile distance for A* since the grid allows diagonal movement
export function AstarHeuristic(a, b) {
    return octileDistance(a, b);
}

export function dijkastraHeuristic(a, b) {
    return 0; // Dijkstra's algorithm uses no heuristic
}

export function greedyHeuristic(a, b) {
    return diagonalDistance(a, b);
}

// Export Pathfinder class


export class Pathfinder {
    constructor(grid, heuristicFn) {
        this.grid = grid;
        this.heuristic = heuristicFn;
        this.nodesVisited = 0;
        this.distanceTraveled = 0;
    }

    async findPath(start, end) {
        // Support both synchronous and asynchronous heuristic functions.
        const evalHeuristic = async (a, b) => {
            try {
                const v = this.heuristic(a, b);
                if (v && typeof v.then === 'function') return await v;
                return v;
            } catch (e) {
                // If heuristic throws, fall back to 0
                return 0;
            }
        };

        // Reset stats
        this.nodesVisited = 0;
        this.distanceTraveled = 0;

        // Initialize grid
        for (let i = 0; i < this.grid.rows; i++) {
            for (let j = 0; j < this.grid.cols; j++) {
                const cell = this.grid.grid[i][j];
                cell.g = Infinity;
                cell.f = Infinity;
                cell.h = 0;
                cell.parent = null;
                cell.visited = false;
            }
        }

        start.g = 0;
        start.h = await evalHeuristic(start, end);
        start.f = start.h;

        const openSet = [start];
        const closedSet = new Set();
        this.nodesVisited = 0;
        this.distanceTraveled = 0;

        const getNeighbors = (cell) => {
            let neighbors = [];
            let directions = [
                { row: -1, col: 0 }, { row: 1, col: 0 }, { row: 0, col: -1 }, { row: 0, col: 1 },
                { row: -1, col: -1 }, { row: -1, col: 1 }, { row: 1, col: -1 }, { row: 1, col: 1 }
            ];
            for (let dir of directions) {
                const r = cell.row + dir.row;
                const c = cell.col + dir.col;
                if (r >= 0 && r < this.grid.rows && c >= 0 && c < this.grid.cols) {
                    const neighbor = this.grid.grid[r][c];
                    if (!neighbor.isWall && !closedSet.has(neighbor)) {
                        if (Math.abs(dir.row) === 1 && Math.abs(dir.col) === 1) {
                            // Check for corner cutting
                            if (this.grid.grid[cell.row][cell.col + dir.col].isWall || 
                                this.grid.grid[cell.row + dir.row][cell.col].isWall) {
                                continue;
                            }
                        }
                        neighbors.push(neighbor);
                    }
                }
            }
            return neighbors;
        };

        while (openSet.length > 0) {
            openSet.sort((a, b) => a.f - b.f);
            const current = openSet.shift();
            if (!current.visited) {
                this.nodesVisited++;
                current.visited = true;
            }

            if (current === end) {
                this.distanceTraveled = current.g;
                return {
                    success: true,
                    nodesVisited: this.nodesVisited,
                    distance: this.distanceTraveled,
                    path: this.reconstructPath(end)
                };
            }

            closedSet.add(current);

            for (const neighbor of getNeighbors(current)) {
                if (closedSet.has(neighbor)) continue;

                const dx = Math.abs(neighbor.row - current.row);
                const dy = Math.abs(neighbor.col - current.col);
                const stepCost = (dx === 1 && dy === 1) ? Math.SQRT2 : 1;
                const tempG = current.g + stepCost * (neighbor.weight || 1);

                if (!openSet.includes(neighbor)) {
                    openSet.push(neighbor);
                } else if (tempG >= neighbor.g) {
                    continue;
                }

                neighbor.parent = current;
                neighbor.g = tempG;
                neighbor.h = await evalHeuristic(neighbor, end);
                neighbor.f = neighbor.g + neighbor.h;
            }
        }

        return {
            success: false,
            nodesVisited: this.nodesVisited,
            distance: 0,
            path: null
        };
    }

    // Backwards-compatible run() and runAsync() wrappers expected by other code
    run(start, end, drawCallback = null, doneCallback = null) {
        // Call findPath (which now supports async heuristics)
        this.findPath(start, end).then((res) => {
            try {
                if (typeof drawCallback === 'function') drawCallback(this.grid);
            } catch (e) {
                // ignore draw errors
            }
            if (typeof doneCallback === 'function') doneCallback(res.success, res.nodesVisited, res.distance);
        }).catch((err) => {
            if (typeof doneCallback === 'function') doneCallback(false, this.nodesVisited, 0);
        });
    }

    async runAsync(start, end, drawCallback = null, doneCallback = null) {
        try {
            const res = await this.findPath(start, end);
            try {
                if (typeof drawCallback === 'function') drawCallback(this.grid);
            } catch (e) {}
            if (typeof doneCallback === 'function') doneCallback(res.success, res.nodesVisited, res.distance);
        } catch (e) {
            if (typeof doneCallback === 'function') doneCallback(false, this.nodesVisited, 0);
        }
    }

    reconstructPath(end) {
        const path = [];
        let current = end;
        while (current) {
            path.unshift(current);
            current = current.parent;
        }
        return path;
    }
}
