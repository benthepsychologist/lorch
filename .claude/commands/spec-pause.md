# Spec Pause: Checkpoint Current Progress

Save current execution state and provide resume instructions.

## Steps:

1. **Log pause event:**
   ```bash
   echo "[$(date -Iseconds)] PAUSED at step <current_step>" >> .aip_artifacts/claude-execution.log
   ```

2. **Show execution summary:**
   ```
   ⏸  EXECUTION PAUSED

   AIP: <title>
   Steps completed: N/M
   Last completed: Step N - <description>
   Next step: Step N+1 - <description>

   Resume with:
     /spec-next
   Or:
     Tell me to "continue from step N+1"
   ```

3. **Show what's been done:**
   - List completed steps
   - List approved gates
   - List created/modified files

4. **Save state** to `.aip_artifacts/execution-state.json`:
   ```json
   {
     "aip_id": "...",
     "paused_at": "2025-11-11T19:30:00Z",
     "last_completed_step": "plan-01",
     "next_step_index": 1,
     "completed_steps": ["plan-01"],
     "approved_gates": ["G0: Plan Approval"]
   }
   ```
