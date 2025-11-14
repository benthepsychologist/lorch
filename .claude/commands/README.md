# Claude Code x Specwright Integration

Slash commands for executing AIPs with Claude Code integrated into the spec workflow.

## Available Commands

### `/spec-run`
**Execute complete AIP step-by-step with gate integration**

Loads the current AIP and executes all steps sequentially:
- Shows each step's prompt, commands, and expected outputs
- Asks for permission before executing
- Runs the step (writes code, executes commands)
- Triggers gate reviews via `spec run --step N`
- Logs all activity to audit trail
- Respects Tier A/B/C governance

**Usage:**
```
/spec-run
```

**When to use:** Starting a new AIP execution from step 1

---

### `/spec-status`
**Show current AIP progress and state**

Displays:
- AIP metadata (ID, title, tier, goal)
- Step completion status (✓ done, ○ pending)
- Gate approval status
- Acceptance criteria checklist
- Next recommended action

**Usage:**
```
/spec-status
```

**When to use:** Check progress, see what's done, find next step

---

### `/spec-next`
**Execute the next incomplete step**

Like `/spec-run` but starts from the next incomplete step instead of step 1.
Useful for resuming after a pause or continuing execution.

**Usage:**
```
/spec-next
```

**When to use:** Continue execution after pause, or after completing one step

---

### `/spec-pause`
**Checkpoint progress and get resume instructions**

Saves current state and provides resume commands.
Logs pause event to audit trail.

**Usage:**
```
/spec-pause
```

**When to use:** Need to stop mid-execution, want to save progress

---

## Complete Workflow Example

### 1. Create and compile a spec
```bash
spec create --tier B --title "Add OAuth" --goal "Implement OAuth2 authentication"
# Edit specs/add-oauth.md
spec compile specs/add-oauth.md
```

### 2. Start execution with Claude
```
/spec-run
```

Claude will:
1. Load the AIP
2. Show step 1 details
3. Ask if you want to execute
4. Execute the step (write code, run commands)
5. Trigger gate review: `spec run --step 1`
6. You approve the gate in the interactive UI
7. Move to step 2
8. Repeat...

### 3. Check progress anytime
```
/spec-status
```

### 4. Pause if needed
```
/spec-pause
```

### 5. Resume later
```
/spec-next
```

---

## How It Works

### Execution Flow
```
You: /spec-run
  ↓
Claude: Loads AIP from .specwright.yaml
  ↓
Claude: Shows Step 1 details
  ↓
You: "Yes, execute"
  ↓
Claude: Writes code, runs commands, shows outputs
  ↓
Claude: Calls `spec run --step 1`
  ↓
Spec CLI: Shows gate checklist (interactive)
  ↓
You: Complete checklist, approve gate
  ↓
Spec CLI: Logs approval to .aip_artifacts/{aip-id}/gate_approvals.jsonl
  ↓
Claude: Proceeds to Step 2
  ↓
[Repeat...]
```

### Audit Trail

All activity is logged:

**Execution History** (`.aip_artifacts/execution_history.jsonl`):
- spec_created
- spec_compiled
- execution_started
- execution_completed

**Gate Approvals** (`.aip_artifacts/{aip-id}/gate_approvals.jsonl`):
- Every gate decision with checklist completion
- Reviewer, timestamp, rationale

**Claude Execution** (`.aip_artifacts/claude-execution.log`):
- Step start/complete timestamps
- Manual log of Claude's actions
- Pause/resume events

### Tier-Specific Behavior

**Tier C (Low Risk):**
- Gates auto-approve (logged only)
- Fast execution
- Minimal ceremony

**Tier B (Moderate Risk):**
- Gates require interactive approval
- Checklists must be completed
- Full audit trail

**Tier A (High Risk):**
- Same as Tier B but more gates
- Stricter approval process
- Cannot skip gates

---

## Integration with Spec CLI

These commands bridge Claude Code with the spec tool:

| Claude Action | Spec CLI Integration | Audit Log |
|--------------|---------------------|-----------|
| Load AIP | Uses `.specwright.yaml` | ✓ |
| Execute step | Follows prompt/commands | Manual log |
| Trigger gate | Calls `spec run --step N` | ✓ gate_approvals.jsonl |
| Complete step | Manual logging | Manual log |

### Future: Native CLI Integration (Phase 2)

When these commands are added to spec CLI:
```bash
spec step-start <step-id>
spec step-complete <step-id> --notes "..."
spec gate-approve <gate-ref> --reviewer "claude-code"
```

The slash commands will automatically use them for better audit logging.

---

## Tips

1. **Always start with `/spec-status`** to see current state
2. **Let gates block progress** - Don't skip them for Tier A/B
3. **Review outputs before approving gates** - Check files were created correctly
4. **Use `/spec-pause` before context switch** - Saves your progress
5. **Check audit logs** to verify everything is tracked:
   ```bash
   cat .aip_artifacts/execution_history.jsonl | jq .
   cat .aip_artifacts/claude-execution.log
   ```

---

## Troubleshooting

**"Can't find current AIP"**
- Run `spec compile <your-spec>.md` first
- Check `.specwright.yaml` has `current.aip` set

**"Gate approval not working"**
- Make sure you're running `spec run --step N` (not just `spec run`)
- Verify AIP has `gate_review` blocks in the compiled YAML

**"Claude skipped a step"**
- Tell Claude to go back: "Execute step N"
- Or use: "Show me step N details, then execute it"

**"Lost progress after pause"**
- Check `.aip_artifacts/claude-execution.log` to see last completed step
- Resume with: `/spec-next` or "Continue from step N"

---

## Next Steps

After setting up this integration, you can:

1. **Phase 2:** Add native CLI commands (`spec step-start`, etc.)
2. **Phase 3:** Build MCP server for deeper integration
3. **Phase 4:** Add automatic acceptance criteria checking
4. **Phase 5:** Build visual dashboard for AIP execution

For now, the slash commands provide a robust bridge between Claude Code and Specwright! 🚀
