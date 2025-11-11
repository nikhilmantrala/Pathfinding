export class Cell {
    constructor(row, col) {
        this.row = row;
        this.col = col;
        this.value = 0;
        this.f = Infinity;
        this.g = Infinity;
        this.h = 0;
        this.isWall = false;
        this.isStart = false;
        this.isEnd = false;
        this.isPath = false;
        this.isVisited = false;
        this.weight = 1.0;
        this.parent = null;
    }

    get color() {
        if (this.isStart) return '#00ff00';  // Green for start
        if (this.isEnd) return '#ff0000';    // Red for end
        if (this.isWall) return '#000000';   // Black for walls
        if (this.isPath) return '#0000ff';   // Blue for path
        if (this.isVisited) return '#aaaaff'; // Light blue for visited
        return '#ffffff';                     // White for empty cells
    }

    set color(value) {
        // Color property is now computed based on cell state
        // This setter is kept for compatibility but doesn't do anything
    }
}

export class Grid {
    constructor(rows, cols, canvas) {
        this.rows = rows;
        this.cols = cols;
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.cellSize = canvas.width / rows;
        this.grid = [];
        this.start = null;
        this.end = null;
        this.init();
    }
    init() {
        this.grid = [];
        for (let i = 0; i < this.rows; i++) {
            let row = [];
            for (let j = 0; j < this.cols; j++) {
                row.push(new Cell(i, j));
            }
            this.grid.push(row);
        }
        this.start = null;
        this.end = null;
    }
    draw(gridToDraw = this.grid) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                let cell = gridToDraw[i][j];
                this.ctx.fillStyle = cell.color;
                this.ctx.fillRect(j * this.cellSize, i * this.cellSize, this.cellSize, this.cellSize);
                this.ctx.strokeStyle = '#e2e8f0';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(j * this.cellSize, i * this.cellSize, this.cellSize, this.cellSize);
            }
        }
    }
    getCell(row, col) {
        return this.grid[row][col];
    }
    resetColors() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                const cell = this.grid[i][j];
                if (!cell.isWall && cell !== this.start && cell !== this.end) {
                    cell.color = 'white';
                }
            }
        }
    }
}

export function cloneGrid(grid) {
    //deep clone(completely new instance) the grid for independent runs
    //need to make it so that grids can be run right next to each other with diff algorithms
    return grid.map(row => row.map(cell => {
        const c = new Cell(cell.row, cell.col);
        c.isWall = cell.isWall;
        c.weight = cell.weight;
        c.color = cell.color;
        return c;
    }));
}
