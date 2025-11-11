export class BatchTestConfig {
    static DENSITY_BANDS = {
        LOW: 'Low',
        MEDIUM: 'Medium',
        HIGH: 'High'
    };

    static LAYOUT_FAMILIES = {
        RANDOM: 'Random',
        MAZE: 'Maze',
        CLUSTERED: 'Clustered',
        MIXED: 'Mixed'
    };

    static ALGORITHM_PAIRS = [
        { name: 'A* vs ML', algorithms: ['astar', 'ml'] },
        { name: 'A*-Dynamic vs ML-Dynamic', algorithms: ['astar_dynamic', 'ml_dynamic'] },
        { name: 'A* vs A*-Dynamic', algorithms: ['astar', 'astar_dynamic'] },
        { name: 'ML vs ML-Dynamic', algorithms: ['ml', 'ml_dynamic'] }
    ];

    static SEED_PRESETS = {
        QUICK: { name: 'Quick', seeds: 5 },
        RECOMMENDED: { name: 'Recommended', seeds: 20 },
        ROBUST: { name: 'Robust', seeds: 50 }
    };

    constructor() {
        this.selectedLayouts = new Set(Object.values(BatchTestConfig.LAYOUT_FAMILIES));
        this.selectedDensities = new Set(Object.values(BatchTestConfig.DENSITY_BANDS));
        this.selectedPair = BatchTestConfig.ALGORITHM_PAIRS[0];
        this.seedsPerFamily = BatchTestConfig.SEED_PRESETS.RECOMMENDED.seeds;
        this.pairsPerSeed = 3;
        this.saveInputs = true;
        this.timeoutMs = 10000; // 10 seconds default timeout
    }

    toJSON() {
        return {
            layouts: Array.from(this.selectedLayouts),
            densities: Array.from(this.selectedDensities),
            algorithmPair: this.selectedPair,
            seedsPerFamily: this.seedsPerFamily,
            pairsPerSeed: this.pairsPerSeed,
            saveInputs: this.saveInputs,
            timeoutMs: this.timeoutMs
        };
    }

    static fromJSON(json) {
        const config = new BatchTestConfig();
        config.selectedLayouts = new Set(json.layouts);
        config.selectedDensities = new Set(json.densities);
        config.selectedPair = json.algorithmPair;
        config.seedsPerFamily = json.seedsPerFamily;
        config.pairsPerSeed = json.pairsPerSeed;
        config.saveInputs = json.saveInputs;
        config.timeoutMs = json.timeoutMs;
        return config;
    }
}
