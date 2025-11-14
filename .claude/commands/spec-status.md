# Spec Status: Show Current AIP Progress

Display the status of the current Agentic Implementation Plan.

## Steps:

1. **Load config:**
   ```bash
   cat .specwright.yaml
   ```

2. **Get current AIP path** from `current.aip`

3. **Parse the AIP:**
   ```bash
   cat <aip-path>
   ```

4. **Display formatted status:**
   ```
   ╔════════════════════════════════════════════════════════════
   ║ CURRENT AIP STATUS
   ╠════════════════════════════════════════════════════════════
   ║ AIP ID: <aip_id>
   ║ Title: <title>
   ║ Tier: <tier>
   ║ Goal: <goal>
   ╠════════════════════════════════════════════════════════════
   ║ STEPS:
   ║
   ║  1. [✓] <step1_description> (plan-01)
   ║  2. [ ] <step2_description> (code-01) ← NEXT
   ║  3. [ ] <step3_description> (test-01)
   ║  ...
   ╠════════════════════════════════════════════════════════════
   ║ GATES:
   ║  G0: Plan Approval - ✓ Approved
   ║  G1: Code Review - Pending
   ║  G2: Pre-Release - Not started
   ╠════════════════════════════════════════════════════════════
   ║ ACCEPTANCE CRITERIA:
   ║  1. [ ] <criterion1>
   ║  2. [ ] <criterion2>
   ║  3. [✓] <criterion3> (if verifiable)
   ╚════════════════════════════════════════════════════════════
   ```

5. **Check audit logs:**
   - Read `.aip_artifacts/claude-execution.log` to see which steps are complete
   - Read `.aip_artifacts/<aip-id>/gate_approvals.jsonl` to see gate statuses
   - Show last activity timestamp

6. **Suggest next action:**
   - "Next: Execute step N with /spec-run"
   - "Or: Resume with: Tell me to continue from step N"
