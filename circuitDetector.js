/**
 * Circuit Detector Module
 * Detects when virtual persons form a circuit by holding hands
 */

export class CircuitDetector {
    constructor(distanceThreshold = 0.30, debounceFrames = 10) {
        this.distanceThreshold = distanceThreshold;
        this.debounceFrames = debounceFrames;
        this.closedCount = 0;
        this.openCount = 0;
        this.currentState = false;
    }

    /**
     * Calculate Euclidean distance between two points
     * @param {Object} p1 - Point 1 {x, y}
     * @param {Object} p2 - Point 2 {x, y}
     * @returns {number} Distance
     */
    calculateDistance(p1, p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    /**
     * Check if all persons have hands raised
     * @param {Array} persons - Array of person objects
     * @returns {boolean} True if all hands are raised
     */
    allHandsRaised(persons) {
        return persons.every(person => {
            const leftRaised = person.leftWrist.y < person.leftShoulder.y;
            const rightRaised = person.rightWrist.y < person.rightShoulder.y;
            return leftRaised && rightRaised;
        });
    }

    /**
     * Check if circuit is closed (instant, no debouncing)
     * @param {Array} persons - Array of person objects
     * @returns {Object} { isClosed, avgDistance, distances }
     */
    checkCircuit(persons) {
        if (!persons || persons.length < 2) {
            return { isClosed: false, avgDistance: 1.0, distances: [] };
        }

        // Check if all hands are raised
        if (!this.allHandsRaised(persons)) {
            return { isClosed: false, avgDistance: 1.0, distances: new Array(persons.length).fill(1.0) };
        }

        const distances = [];
        let totalDistance = 0;

        // Calculate distances between adjacent persons
        for (let i = 0; i < persons.length; i++) {
            const currentPerson = persons[i];
            const nextPerson = persons[(i + 1) % persons.length]; // Wrap around to first person

            // Current person's right hand to next person's left hand
            const dist = this.calculateDistance(currentPerson.rightWrist, nextPerson.leftWrist);
            distances.push(dist);
            totalDistance += dist;
        }

        const avgDistance = totalDistance / persons.length;

        // Check if all distances are below threshold
        const allClose = distances.every(d => d < this.distanceThreshold);

        return { isClosed: allClose, avgDistance, distances };
    }

    /**
     * Update circuit state with debouncing
     * @param {Array} persons - Array of 3 person objects
     * @returns {Object} { state, avgDistance, distances, stateChanged }
     */
    update(persons) {
        const { isClosed, avgDistance, distances } = this.checkCircuit(persons);

        let stateChanged = false;

        if (isClosed) {
            this.closedCount++;
            this.openCount = 0;

            if (this.closedCount >= this.debounceFrames && !this.currentState) {
                this.currentState = true;
                stateChanged = true;
                console.log(`🔌 Circuit closed! (consecutive frames: ${this.closedCount})`);
            }
        } else {
            this.openCount++;
            this.closedCount = 0;

            if (this.openCount >= this.debounceFrames && this.currentState) {
                this.currentState = false;
                stateChanged = true;
                console.log(`🔓 Circuit opened (consecutive frames: ${this.openCount})`);
            }
        }

        return {
            state: this.currentState,
            avgDistance,
            distances,
            stateChanged
        };
    }

    /**
     * Get current state
     * @returns {boolean} Current circuit state
     */
    getState() {
        return this.currentState;
    }

    /**
     * Reset the detector
     */
    reset() {
        this.closedCount = 0;
        this.openCount = 0;
        this.currentState = false;
    }
}
