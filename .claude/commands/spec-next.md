# Spec Next: Execute Next Step

Execute the next incomplete step in the current AIP.

## Protocol:

1. **Load current AIP** from `.specwright.yaml`

2. **Determine next step:**
   - Check `.aip_artifacts/claude-execution.log` for last completed step
   - Find next step in plan array

3. **Execute that step** using the same protocol as `/spec-run`:
   - Display step details
   - Ask for confirmation
   - Execute the prompt
   - Run commands
   - Verify outputs
   - Trigger gate if present: `spec run --step N`

4. **When complete:**
   - Show what was done
   - Ask: "Continue to next step?"
   - If yes, repeat; if no, show resume command

This is essentially `/spec-run` but starting from the next incomplete step instead of step 1.
