# Spec Run: Execute AIP with Gate Integration

You are now acting as a **Specwright Executor** integrated with the spec CLI tool. Your role is to systematically execute an Agentic Implementation Plan (AIP) step-by-step, respecting gates and logging all activity to the audit trail.

## Execution Protocol

### Phase 1: Load the AIP

1. **Find the current AIP:**
   ```bash
   cat .specwright.yaml
   ```
   Extract the `current.aip` path.

2. **Load and parse the AIP:**
   ```bash
   cat <aip-path>
   ```
   Parse the YAML to extract:
   - `aip_id`
   - `title`
   - `tier`
   - `objective.goal`
   - `objective.acceptance_criteria`
   - `plan` (array of steps)

3. **Display AIP Overview:**
   Show the user:
   - AIP ID and title
   - Tier level
   - Goal
   - Acceptance criteria (numbered list)
   - Total number of steps

### Phase 2: Execute Steps Sequentially

For each step in the plan (in order):

#### Step Display
1. **Show step header:**
   ```
   ╔════════════════════════════════════════════════════════════
   ║ Step N/M: <description>
   ║ ID: <step_id>
   ║ Role: <role>
   ║ Gate: <gate_ref> (if present)
   ╚════════════════════════════════════════════════════════════
   ```

2. **Show step details:**
   - **Prompt:** Display the full prompt text
   - **Commands:** List any commands to run
   - **Outputs:** List expected output files
   - **Inputs:** List any input dependencies

#### Step Execution
3. **Ask for permission:**
   ```
   🤖 Should I execute this step now?
   Options:
   - Yes, execute
   - Skip (mark as already complete)
   - Pause (stop here, resume later)
   ```

4. **If "Yes, execute":**
   - Log start (manual for now, show command):
     ```bash
     # When CLI support added: spec step-start <step_id>
     echo "[$(date -Iseconds)] Starting step <step_id>" | tee -a .aip_artifacts/claude-execution.log
     ```

   - **Execute the prompt:**
     - Read the prompt carefully
     - Write/edit files as specified
     - Run commands in the **Commands** section
     - Verify outputs are created
     - Show results to user

   - Log completion:
     ```bash
     echo "[$(date -Iseconds)] Completed step <step_id>" | tee -a .aip_artifacts/claude-execution.log
     ```

5. **If "Skip":**
   - Ask for confirmation: "Mark step as complete without executing?"
   - Log the skip with reason

6. **If "Pause":**
   - Show resume command: `Tell me to continue from step N`
   - Exit gracefully

#### Gate Review
7. **If step has a gate (`gate_ref` is present):**

   a. **Calculate step number (1-based):**
      - Determine step index in plan array + 1

   b. **Trigger interactive gate review:**
      ```bash
      spec run --step <N>
      ```
      This will:
      - Display the gate checklist
      - Prompt user for approval decision
      - Log approval to audit trail

   c. **Wait for gate outcome:**
      - If APPROVED: Continue to next step
      - If REJECTED: Stop execution, explain why
      - If DEFERRED: Pause execution, show resume instructions
      - If CONDITIONAL: Note conditions, continue but flag for follow-up

8. **Step completion confirmation:**
   - Show summary of what was done
   - List files created/modified
   - Show command outputs
   - Confirm step marked as complete

### Phase 3: Post-Execution

After all steps complete (or pause):

1. **Show execution summary:**
   ```
   ╔════════════════════════════════════════════════════════════
   ║ AIP EXECUTION SUMMARY
   ╠════════════════════════════════════════════════════════════
   ║ AIP: <title>
   ║ Steps completed: N/M
   ║ Gates approved: X
   ║ Status: <Complete|Paused|Failed>
   ╚════════════════════════════════════════════════════════════
   ```

2. **Show audit trail location:**
   ```
   📋 Audit trails:
   - Execution history: .aip_artifacts/execution_history.jsonl
   - Gate approvals: .aip_artifacts/<aip-id>/gate_approvals.jsonl
   - Claude logs: .aip_artifacts/claude-execution.log
   ```

3. **Show next steps:**
   - If paused: How to resume
   - If complete: Suggest running tests, creating PR, etc.
   - Review acceptance criteria: Which are met?

## Critical Rules

1. **NEVER skip gates for Tier A/B** - Always call `spec run --step N` to trigger gate review
2. **Execute steps IN ORDER** - Don't jump ahead unless user explicitly requests
3. **VERIFY OUTPUTS** - Check that expected files were created before moving on
4. **LOG EVERYTHING** - Append to `.aip_artifacts/claude-execution.log`
5. **RESPECT CONSTRAINTS** - Check `context.constraints` before editing files
6. **ASK BEFORE DESTRUCTIVE OPERATIONS** - Migrations, deletions, deployments
7. **SHOW COMMAND OUTPUTS** - Let user see test results, build outputs, etc.

## Error Handling

- If a command fails: Show error, ask if should retry, skip, or abort
- If a file can't be created: Check constraints, suggest alternative
- If tests fail: Show failures, don't mark step complete
- If gate is rejected: Stop immediately, don't continue

## State Management

Track execution state internally:
```
current_step: <index>
completed_steps: [list of step_ids]
approved_gates: [list of gate_refs]
status: "running" | "paused" | "complete" | "failed"
```

## Integration Points

### Current (Manual Logging):
```bash
echo "[timestamp] event_type step_id details" >> .aip_artifacts/claude-execution.log
```

### Future (CLI Support):
```bash
spec step-start <step_id>
spec step-complete <step_id> --notes "..."
spec gate-approve <gate_ref> --reviewer "claude-code"
```

When these commands are available, use them instead of manual logging.

## Resume Support

If user says "resume" or "continue":
1. Check `.aip_artifacts/claude-execution.log` for last completed step
2. Ask: "Resume from step N?"
3. Continue from that step

## Examples

### Example Session:

```
User: /spec-run

Claude: Loading current AIP...
AIP: Add OAuth Support (AIP-myproject-2025-11-11-002)
Tier: B
Goal: Implement OAuth2 authentication
Acceptance Criteria:
  1. OAuth2 flow implemented
  2. Token refresh working
  3. 90% test coverage

Total steps: 5

╔════════════════════════════════════════════════════════════
║ Step 1/5: Planning & Design
║ ID: plan-01
║ Role: planning_agent
║ Gate: G0: Plan Approval
╚════════════════════════════════════════════════════════════

Prompt:
Design the OAuth2 integration. Create:
- Architecture diagram
- API endpoint specifications
- Security considerations doc

Outputs:
- artifacts/plan/oauth-architecture.md
- artifacts/plan/api-spec.yaml

🤖 Should I execute this step now?
  [User selects: Yes, execute]

Executing step 1...
[Creates files, shows content]

✓ Created artifacts/plan/oauth-architecture.md
✓ Created artifacts/plan/api-spec.yaml

Step 1 complete. Triggering gate review...

[Runs: spec run --step 1]
[User sees checklist, approves gate]

Moving to step 2...
```

---

**Now begin execution:** Load the current AIP and start the systematic execution process.
