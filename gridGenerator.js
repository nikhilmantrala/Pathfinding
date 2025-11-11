export class GridGenerator {
    constructor(grid) {
        this.grid = grid;
    }

    generateMaze() {
        this.clearGrid();

        // Initialize maze with a grid of walls, leaving every other cell as a passage
        for (let y = 0; y < this.grid.rows; y++) {
            for (let x = 0; x < this.grid.cols; x++) {
                if (y % 2 === 0 && x % 2 === 0) {
                    this.grid.grid[y][x].isWall = false; // Create passage points
                } else {
                    this.grid.grid[y][x].isWall = true;
                }
            }
        }

        // Helper to get unvisited neighbors
        const getUnvisitedNeighbors = (x, y, visited) => {
            const neighbors = [];
            const directions = [
                { dx: 0, dy: -2 }, // North
                { dx: 2, dy: 0 },  // East
                { dx: 0, dy: 2 },  // South
                { dx: -2, dy: 0 }  // West
            ];

            for (const dir of directions) {
                const newX = x + dir.dx;
                const newY = y + dir.dy;
                if (newX >= 0 && newX < this.grid.cols && 
                    newY >= 0 && newY < this.grid.rows && 
                    !visited.has(`${newX},${newY}`)) {
                    neighbors.push({ x: newX, y: newY, dx: dir.dx/2, dy: dir.dy/2 });
                }
            }
            return neighbors;
        };

        // Use a modified randomized DFS to create the maze
        const visited = new Set();
        const stack = [];
        const start = { x: 0, y: 0 };
        visited.add(`${start.x},${start.y}`);
        stack.push(start);

        while (stack.length > 0) {
            const current = stack[stack.length - 1];
            const neighbors = getUnvisitedNeighbors(current.x, current.y, visited);

            if (neighbors.length > 0) {
                // Randomly select a neighbor
                const next = neighbors[Math.floor(Math.random() * neighbors.length)];
                
                // Create a passage by removing the wall between current and next
                const wallX = current.x + next.dx;
                const wallY = current.y + next.dy;
                this.grid.grid[wallY][wallX].isWall = false;

                // Mark the new cell as visited and add it to the stack
                visited.add(`${next.x},${next.y}`);
                stack.push(next);
            } else {
                stack.pop();
            }
        }

        // Add additional passages to make the maze more open
        const addExtraPassages = () => {
            const numExtra = Math.floor((this.grid.rows + this.grid.cols) / 4);
            for (let i = 0; i < numExtra; i++) {
                const x = Math.floor(Math.random() * (this.grid.cols - 2)) + 1;
                const y = Math.floor(Math.random() * (this.grid.rows - 2)) + 1;
                
                // Create a small opening
                this.grid.grid[y][x].isWall = false;
                
                // Randomly remove adjacent walls to create wider passages
                if (Math.random() < 0.5 && x > 0) this.grid.grid[y][x-1].isWall = false;
                if (Math.random() < 0.5 && x < this.grid.cols-1) this.grid.grid[y][x+1].isWall = false;
                if (Math.random() < 0.5 && y > 0) this.grid.grid[y-1][x].isWall = false;
                if (Math.random() < 0.5 && y < this.grid.rows-1) this.grid.grid[y+1][x].isWall = false;
            }
        };

        // Add extra passages for better connectivity
        addExtraPassages();

        // Ensure edges are accessible
        for (let i = 0; i < this.grid.rows; i++) {
            if (Math.random() < 0.3) this.grid.grid[i][0].isWall = false;
            if (Math.random() < 0.3) this.grid.grid[i][this.grid.cols-1].isWall = false;
        }
        for (let j = 0; j < this.grid.cols; j++) {
            if (Math.random() < 0.3) this.grid.grid[0][j].isWall = false;
            if (Math.random() < 0.3) this.grid.grid[this.grid.rows-1][j].isWall = false;
        }
    }

    generateClusteredObstacles() {
        this.clearGrid();
        const numClusters = Math.floor(Math.random() * 3) + 3; // 3-5 clusters

        // Keep track of cluster centers to ensure they're not too close
        const clusterCenters = [];
        const minDistanceBetweenClusters = 5;

        for (let attempts = 0; attempts < 20 && clusterCenters.length < numClusters; attempts++) {
            const centerX = Math.floor(Math.random() * this.grid.cols);
            const centerY = Math.floor(Math.random() * this.grid.rows);

            // Check if this center is far enough from other centers
            let tooClose = false;
            for (const center of clusterCenters) {
                const dx = centerX - center.x;
                const dy = centerY - center.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < minDistanceBetweenClusters) {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose) {
                clusterCenters.push({ x: centerX, y: centerY });
                const size = Math.floor(Math.random() * 3) + 2; // 2-4 size (slightly smaller)

                for (let y = -size; y <= size; y++) {
                    for (let x = -size; x <= size; x++) {
                        const newX = centerX + x;
                        const newY = centerY + y;
                        if (newX >= 0 && newX < this.grid.cols && 
                            newY >= 0 && newY < this.grid.rows) {
                            
                            // Calculate distance from cluster center
                            const distance = Math.sqrt(x * x + y * y);
                            
                            // Higher chance of walls near center, lower at edges
                            const wallChance = Math.max(0, 0.8 - (distance / size) * 0.3);
                            if (Math.random() < wallChance) {
                                this.grid.grid[newY][newX].isWall = true;
                            }
                        }
                    }
                }
            }
        }
    }

    generateRandom(densityBand = 'Medium') {
        this.clearGrid();
        // Pre-generate random values for better performance
        const walls = new Float32Array(this.grid.rows * this.grid.cols);
        for (let i = 0; i < walls.length; i++) {
            walls[i] = Math.random();
        }
        
        let wallProbability;
        switch(densityBand) {
            case 'Low':
                wallProbability = 0.2;
                break;
            case 'High':
                wallProbability = 0.4;
                break;
            default: // Medium
                wallProbability = 0.3;
        }
        
        let idx = 0;
        for (let i = 0; i < this.grid.rows; i++) {
            for (let j = 0; j < this.grid.cols; j++) {
                if (walls[idx++] < wallProbability) {
                    this.grid.grid[i][j].isWall = true;
                }
            }
        }
    }

    generateRandomCosts() {
        // Pre-generate random numbers for better performance
        const costs = new Float32Array(this.grid.rows * this.grid.cols * 2);
        for (let i = 0; i < costs.length; i++) {
            costs[i] = Math.random();
        }
        
        let idx = 0;
        for (let i = 0; i < this.grid.rows; i++) {
            for (let j = 0; j < this.grid.cols; j++) {
                if (!this.grid.grid[i][j].isWall && costs[idx++] < 0.3) {
                    const cost = 1 + costs[idx++] * 2; // cost between 1-3
                    this.grid.grid[i][j].cost = cost;
                    this.grid.grid[i][j].weight = cost;
                }
            }
        }
    }

    clearGrid() {
        for (let i = 0; i < this.grid.rows; i++) {
            for (let j = 0; j < this.grid.cols; j++) {
                this.grid.grid[i][j].isWall = false;
                this.grid.grid[i][j].cost = 1;
                this.grid.grid[i][j].weight = 1;
            }
        }
    }

    generate(type, densityBand = 'Medium') {
        switch(type.toLowerCase()) {
            case 'maze':
                this.generateMaze();
                break;
            case 'clustered':
                this.generateClusteredObstacles();
                // Adjust cluster size based on density
                if (densityBand === 'Low') {
                    this.clearRandomWalls(0.3); // Remove 30% of walls
                } else if (densityBand === 'High') {
                    this.addRandomWalls(0.2); // Add 20% more walls
                }
                break;
            case 'random':
                this.generateRandom(densityBand);
                break;
            case 'mixed':
                // Divide the grid into sections
                const sections = this.generateMixedLayout(densityBand);
                
                // Apply different patterns to each section
                for (const section of sections) {
                    this.generateSectionPattern(section, densityBand);
                }
                
                // Ensure connectivity between sections
                this.connectSections(sections);
                break;
        }
        this.generateRandomCosts();
    }
    
    clearRandomWalls(percentage) {
        for (let i = 0; i < this.grid.rows; i++) {
            for (let j = 0; j < this.grid.cols; j++) {
                if (this.grid.grid[i][j].isWall && Math.random() < percentage) {
                    this.grid.grid[i][j].isWall = false;
                }
            }
        }
    }
    
    addRandomWalls(percentage) {
        for (let i = 0; i < this.grid.rows; i++) {
            for (let j = 0; j < this.grid.cols; j++) {
                if (!this.grid.grid[i][j].isWall && Math.random() < percentage) {
                    this.grid.grid[i][j].isWall = true;
                }
            }
        }
    }

    generateMixedLayout(densityBand) {
        this.clearGrid();
        
        // Create 2-4 sections
        const numSections = Math.floor(Math.random() * 3) + 2;
        const sections = [];
        
        if (Math.random() < 0.5) {
            // Horizontal sections
            const sectionHeight = Math.floor(this.grid.rows / numSections);
            for (let i = 0; i < numSections; i++) {
                const startY = i * sectionHeight;
                const endY = (i === numSections - 1) ? this.grid.rows : (i + 1) * sectionHeight;
                sections.push({
                    x1: 0,
                    y1: startY,
                    x2: this.grid.cols,
                    y2: endY,
                    type: ['maze', 'clustered', 'random'][Math.floor(Math.random() * 3)]
                });
            }
        } else {
            // Vertical sections
            const sectionWidth = Math.floor(this.grid.cols / numSections);
            for (let i = 0; i < numSections; i++) {
                const startX = i * sectionWidth;
                const endX = (i === numSections - 1) ? this.grid.cols : (i + 1) * sectionWidth;
                sections.push({
                    x1: startX,
                    y1: 0,
                    x2: endX,
                    y2: this.grid.rows,
                    type: ['maze', 'clustered', 'random'][Math.floor(Math.random() * 3)]
                });
            }
        }
        
        return sections;
    }

    generateSectionPattern(section, densityBand) {
        // Create a subgrid view
        const subgrid = {
            grid: this.grid.grid,
            rows: section.y2 - section.y1,
            cols: section.x2 - section.x1
        };

        // Save current grid
        const tempGrid = this.grid;
        
        // Create offset wrapper for the section
        this.grid = {
            grid: Array(subgrid.rows).fill().map((_, i) => 
                Array(subgrid.cols).fill().map((_, j) => 
                    tempGrid.grid[i + section.y1][j + section.x1]
                )
            ),
            rows: subgrid.rows,
            cols: subgrid.cols
        };

        // Generate pattern based on type
        switch(section.type) {
            case 'maze':
                this.generateMaze();
                // Add some random openings for better connectivity
                if (densityBand === 'Low') {
                    this.clearRandomWalls(0.4);
                } else if (densityBand === 'Medium') {
                    this.clearRandomWalls(0.2);
                }
                break;
            case 'clustered':
                this.generateClusteredObstacles();
                // Adjust density
                if (densityBand === 'Low') {
                    this.clearRandomWalls(0.3);
                } else if (densityBand === 'High') {
                    this.addRandomWalls(0.15);
                }
                break;
            case 'random':
                this.generateRandom(densityBand);
                break;
        }

        // Restore original grid
        this.grid = tempGrid;
    }

    connectSections(sections) {
        // Create passages between sections
        for (let i = 0; i < sections.length - 1; i++) {
            const current = sections[i];
            const next = sections[i + 1];
            
            // Determine if sections are adjacent horizontally or vertically
            if (current.x2 === next.x1) {
                // Vertical boundary
                const passages = Math.max(3, Math.floor((current.y2 - current.y1) / 5));
                for (let p = 0; p < passages; p++) {
                    const y = current.y1 + Math.floor((current.y2 - current.y1) * (p + 1) / (passages + 1));
                    // Create a small passage
                    for (let x = -1; x <= 1; x++) {
                        if (current.x2 + x >= 0 && current.x2 + x < this.grid.cols) {
                            this.grid.grid[y][current.x2 + x].isWall = false;
                            // Clear adjacent cells for better connectivity
                            if (y > 0) this.grid.grid[y-1][current.x2 + x].isWall = false;
                            if (y < this.grid.rows - 1) this.grid.grid[y+1][current.x2 + x].isWall = false;
                        }
                    }
                }
            } else if (current.y2 === next.y1) {
                // Horizontal boundary
                const passages = Math.max(3, Math.floor((current.x2 - current.x1) / 5));
                for (let p = 0; p < passages; p++) {
                    const x = current.x1 + Math.floor((current.x2 - current.x1) * (p + 1) / (passages + 1));
                    // Create a small passage
                    for (let y = -1; y <= 1; y++) {
                        if (current.y2 + y >= 0 && current.y2 + y < this.grid.rows) {
                            this.grid.grid[current.y2 + y][x].isWall = false;
                            // Clear adjacent cells for better connectivity
                            if (x > 0) this.grid.grid[current.y2 + y][x-1].isWall = false;
                            if (x < this.grid.cols - 1) this.grid.grid[current.y2 + y][x+1].isWall = false;
                        }
                    }
                }
            }
        }

        // Add some random passages throughout
        const numExtraPassages = Math.floor((this.grid.rows + this.grid.cols) / 8);
        for (let i = 0; i < numExtraPassages; i++) {
            const x = Math.floor(Math.random() * (this.grid.cols - 2)) + 1;
            const y = Math.floor(Math.random() * (this.grid.rows - 2)) + 1;
            
            // Create a passage
            this.grid.grid[y][x].isWall = false;
            // Clear some adjacent cells for better connectivity
            if (x > 0) this.grid.grid[y][x-1].isWall = false;
            if (x < this.grid.cols - 1) this.grid.grid[y][x+1].isWall = false;
            if (y > 0) this.grid.grid[y-1][x].isWall = false;
            if (y < this.grid.rows - 1) this.grid.grid[y+1][x].isWall = false;
        }
    }
}
