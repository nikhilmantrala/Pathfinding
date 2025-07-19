export class Cell {
    constructor(row, col) {
        this.row = row;
        this.col = col;
        this.value = 0;
        this.f = Infinity;
        this.g = Infinity;
        this.h = 0;
        this.isWall = false;
        this.weight = 1.0;
        this.color = 'white';
        this.parent = null;
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
    //Deep clone(completely new instance) the grid for independent runs
    //Need to make it so that grids can be run right next to each other with diff algorithms
    return grid.map(row => row.map(cell => {
        const c = new Cell(cell.row, cell.col);
        c.isWall = cell.isWall;
        c.weight = cell.weight;
        c.color = cell.color;
        return c;
    }));
}
