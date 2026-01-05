### Final Project Recap: Constraint Bootstrap (v1.1)

This refined recap incorporates the tight semantic corrections provided by Q to ensure the system's invariants are perfectly documented for future development.

#### 1. Core Architecture: The Truth/Eligibility Invariant
The agent's "mental model" consists of **Handles** (Input-Output mappings).
*   **Dual-Score Gating**:
    *   **Eligibility**: Grows via exposure and environmental pressure (e.g., Silence Penalty). It represents familiarity.
    *   **Truth**: **Must be earned.** It only grows during trainable correction events (where an oracle confirms a SPEAK prediction).
    *   **Strength**: Defined as `min(Eligibility, Truth)`. This ensures the agent never speaks based on unverified familiarity or unexposed truth.
*   **SPEAK Gating**: Gated primarily by **Strength**. While `strength` (min of E and T) serves as the master gate, `truth_min_to_speak` is maintained as an **extra safety clamp** to ensure a hard floor on evidence-backed correctness regardless of eligibility levels.

#### 2. The 4-Lane System (Routing Rules)
The system distinguishes between four non-overlapping lanes:
*   **SILENT (Empty-State)**: The base state when **zero matching handles** exist in the registry for the current input.
*   **NA (Pre-Eligible)**: Matches exist in the registry, but none have reached the `eligibility_min_to_consider` threshold to be "known."
*   **QUESTION (The Knowledge Bridge)**:
    1.  **Weak Knowledge**: Known candidates exist but fail the Strength/Truth gates for SPEAKing.
    2.  **Conflict**: Two or more strong candidates exist and their strengths are within the `conflict_margin`. **Conflict is computed exclusively among the top eligible competitors** (after filtering for eligibility and Top-K).
*   **SPEAK**: The top candidate passes all gates and wins the competition.

#### 3. QUESTION Semantic & Scoring Invariant
*   **Utility/Loss Placeholder**: In error metrics, a QUESTION incurs a **fixed loss/cost of 0.5**. This is a placeholder for utility/loss math, not a correctness score. It represents a controlled cost—preferable to a wrong SPEAK but more "expensive" than a correct SPEAK. It is excluded from action-space accuracy math.

#### 4. Surprise (De) & The Obsession Guardrail
*   **Surprise (De)**: Triggered when prediction error exceeds the `surprise_threshold`. It is a behavioral event, not a distinct output lane.
*   **The Guardrail**: Surprise triggers an **Obsession Loop** (repeating the input). **Truth is never "gifted" during this loop**; handles only update their truth weights if a standard trainable correction event occurs during the repeats.

#### 5. Training Dynamics
*   **Aggressive Training**: Leverages **Boundary Drills** (uncertainty-driven sampling) and a **Silence Penalty**.
*   **Silence Penalty Guardrail**: The penalty nudges the **eligibility** (gates/calibration) of missed mappings but never modifies truth weights. This prevents the agent from "lying" under pressure.

#### 6. Analysis KPI: Response Efficiency
Performance is primarily evaluated via **Response Efficiency**, which measures accuracy exclusively on events where the agent chose to enter the SPEAK lane, ensuring the correctness metric is not diluted by uncertainty-admitting states.