# Specwright Guide: Writing Agentic Implementation Plans (AIPs)

## Overview

Specwright helps you create **Agentic Implementation Plans (AIPs)** - structured specifications that guide AI agents through complex software engineering tasks with clear governance and human oversight.

An AIP is a living document that:
- Defines **what** needs to be built (objective & acceptance criteria)
- Specifies **how** to build it (step-by-step plan with gates)
- Establishes **governance** (risk tier, constraints, rollback procedures)

## Risk Tiers

Specwright uses three risk tiers that determine governance requirements:

### Tier C (Low Risk)
- **Use for:** Small features, bug fixes, refactoring, documentation
- **Constraints:** No protected path edits, quick validation only
- **Gates:** Minimal - plan approval, pre-release checks
- **Steps:** 4-5 steps, minimal ceremony
- **Example:** "Add logging to user authentication flow"

### Tier B (Moderate Risk)
- **Use for:** New features, API changes, moderate complexity
- **Constraints:** Protected path edits with review, full testing
- **Gates:** More frequent - plan, design, implementation, pre-release
- **Steps:** 6-8 steps with design docs
- **Example:** "Implement user role-based permissions"

### Tier A (High Risk)
- **Use for:** Core system changes, migrations, security-critical features
- **Constraints:** Strict protected paths, full audit trail, staged rollout
- **Gates:** Comprehensive - every major decision point needs approval
- **Steps:** 8-12 steps with extensive documentation
- **Example:** "Migrate authentication system to OAuth2"

## Anatomy of a Spec (Markdown Format)

```markdown
---
version: "0.1"
tier: C
title: Add User Avatar Support
owner: alice
goal: Allow users to upload and display profile avatars
labels: [feature, frontend, backend]
orchestrator_contract: "standard"
repo:
  working_branch: "feat/user-avatars"
---

# Add User Avatar Support

## Objective

> Allow users to upload and display profile avatars

### Background

Currently, users are represented by initials only. User feedback indicates
that profile pictures would improve recognition and engagement.

### Acceptance Criteria

- Users can upload image files (PNG, JPG, max 5MB)
- Avatars display in user profile and navigation bar
- Default avatar shown for users without uploads
- Images are resized to 200x200px thumbnails
- 95% test coverage on new avatar code

### Constraints

- No edits under protected paths (`src/core/**`, `infra/**`)
- Must work with existing authentication system
- Image storage must comply with GDPR (user data deletion)

## Plan

### Step 1: Design Phase [G0: Design Approval]

**Prompt:**

Design the avatar upload system. Specify:
- API endpoints (upload, retrieve, delete)
- Database schema changes (users table avatar_url field)
- Storage solution (S3 bucket structure)
- Security considerations (file validation, size limits)

Produce a design doc in `docs/design/avatars.md`.

**Outputs:**
- `docs/design/avatars.md`

---

### Step 2: Backend Implementation [G1: Code Review]

**Prompt:**

Implement the avatar upload API:
1. Add `avatar_url` column to users table (migration)
2. Create `/api/users/avatar` endpoints (POST, GET, DELETE)
3. Implement S3 upload with image validation
4. Add unit tests for avatar service

**Commands:**

```bash
# Run database migration
python manage.py db migrate -m "Add avatar_url to users"

# Run tests
pytest tests/api/test_avatars.py -v
```

**Outputs:**
- `backend/models/user.py` (updated)
- `backend/api/avatars.py` (new)
- `backend/services/avatar_service.py` (new)
- `tests/api/test_avatars.py` (new)
- Migration file

---

### Step 3: Frontend Implementation [G2: Manual Testing]

**Prompt:**

Implement the avatar UI:
1. Add avatar upload component to profile page
2. Display avatars in navigation bar
3. Handle upload progress and errors
4. Add default avatar fallback

**Commands:**

```bash
npm run test
npm run build
```

**Outputs:**
- `frontend/components/AvatarUpload.tsx` (new)
- `frontend/components/UserAvatar.tsx` (new)
- `frontend/pages/Profile.tsx` (updated)

---

### Step 4: Integration Testing [G3: Pre-Release]

**Prompt:**

Run end-to-end tests for avatar workflow:
1. User uploads avatar
2. Avatar appears in profile
3. Avatar appears in navigation
4. User can delete avatar

**Commands:**

```bash
npm run test:e2e
pytest tests/ --cov=backend
```

**Outputs:**
- Test coverage report
- E2E test results

---

### Step 5: Documentation & Rollback [G4: Final Approval]

**Prompt:**

Create deployment and rollback documentation.

**Outputs:**
- `docs/deployment/avatar-feature.md`
- `docs/rollback/avatar-feature.md`

## Orchestrator

**State Machine:** Standard (pending → running → awaiting_human → succeeded/failed)

**Tools:** bash, pytest, npm, git

**Models:** (to be filled by defaults)

## Repository

**Branch:** `feat/user-avatars`

**Merge Strategy:** squash
```

## Writing Effective Steps

Each step should follow this pattern:

### 1. Clear Prompt
- Start with an imperative verb (Implement, Design, Test, Deploy)
- Be specific about what the agent should create
- Include success criteria
- Reference existing code/docs when relevant

**Good:** "Implement user avatar upload with file validation (PNG/JPG, max 5MB), S3 storage, and database persistence."

**Bad:** "Add avatar stuff."

### 2. Commands (Optional)
- Include actual shell commands the agent should run
- Use bash code blocks
- Add commands for testing, building, validation

```bash
# Good example
pytest tests/api/test_avatars.py -v
ruff check src/
```

### 3. Outputs
- List all files that should be created or modified
- Be specific with paths
- Include test files and documentation

```
**Outputs:**
- `backend/api/avatars.py` (new)
- `backend/models/user.py` (updated)
- `tests/api/test_avatars.py` (new)
- `docs/api/avatars.md` (new)
```

### 4. Gates
- Use `[G0: Description]`, `[G1: Description]`, etc. in step titles
- Gates are human approval checkpoints
- More gates = more oversight (Tier A has more gates than Tier C)
- Common gates:
  - **G0: Plan Approval** - Initial planning complete
  - **G1: Design Review** - Architecture/design validated
  - **G2: Code Review** - Implementation ready
  - **G3: Pre-Release** - Testing complete
  - **G4: Final Approval** - Ready to deploy

#### Gate Review Checklists (Tier A/B)

For Tier A and B specs, you can add detailed review checklists to gate steps. These checklists are displayed interactively during `spec run` and require approval before proceeding.

**Understanding Gate Structure:**

Gates exist in TWO representations:
1. **Markdown Format (Human-Authored)**: Gate review blocks embedded in steps using HTML comment markers
2. **YAML Format (Compiled)**: Separate gate items with `type: "gate"` in the compiled AIP

Both represent the same validation checkpoints - the markdown format is for humans to write, the YAML format is for machines to execute.

**When to Add Gate Reviews:**
- **Always for Tier A/B**: Add gate review blocks to steps with gate references (`[G0: ...]`, `[G1: ...]`, etc.)
- **Optional for Tier C**: Can be added but will auto-approve

**Where to Place:**
- After `**Validation:**` section (or after `**Outputs:**` if no validation section)
- Before the next step header
- Must be within the step content (between step headers)

**Exact Format (Copy This Template):**
```markdown
### Step N: Step Title [G0: Plan Approval]

**Prompt:**
Your step prompt here...

**Commands:**
```bash
commands here
```

**Validation:**

Before proceeding to G0 gate review, verify:
- ✓ All commands pass without errors
- ✓ All output artifacts generated
- ✓ No blocked paths modified

**Outputs:**
- `file1.py`
- `file2.md`

<!-- GATE_REVIEW_START -->
#### Gate Review Checklist

##### Category Name 1
- [ ] Checklist item 1
- [ ] Checklist item 2
- [ ] Checklist item 3

##### Category Name 2
- [ ] Checklist item 4
- [ ] Checklist item 5

#### Approval Decision
- [ ] APPROVED
- [ ] APPROVED WITH CONDITIONS: ___
- [ ] REJECTED: ___
- [ ] DEFERRED: ___

**Approval Metadata:**
- Reviewer: ___
- Date: ___
- Rationale: ___
<!-- GATE_REVIEW_END -->
```

**CRITICAL Rules for LLMs:**

1. **HTML Comment Markers are REQUIRED**: Must use `<!-- GATE_REVIEW_START -->` and `<!-- GATE_REVIEW_END -->` exactly as shown
2. **Heading Levels Matter**:
   - `#### Gate Review Checklist` (4 hashes)
   - `##### Category Name` (5 hashes for categories)
   - `#### Approval Decision` (4 hashes)
3. **Checkbox Format**: Must be `- [ ]` with a space between brackets
4. **Section Order**: Checklist categories first, then Approval Decision, then Approval Metadata
5. **No Extra Content**: Don't add text between sections or modify the structure

**Validation Checkpoints vs Gate Reviews:**

Understanding the difference between lightweight validation and formal gate approval:

**Validation Checkpoints** (Automated, Lightweight):
- Run DURING step execution
- Automated checks: tests pass, files generated, commands succeed
- No human approval required
- Examples: "Run pytest -q", "Check coverage ≥85%", "Verify artifact created"
- Used in `**Validation:**` section

**Gate Reviews** (Human Approval, Formal):
- Run AFTER step completion
- Requires human review and approval
- Interactive checklist completion
- Examples: "Architecture sound", "Security review complete", "QA sign-off obtained"
- Used in `<!-- GATE_REVIEW_START -->` block

**Both work together:**
```markdown
**Commands:**
```bash
pytest -q
```

**Validation:** ← Automated checks (no approval)
- ✓ Tests pass
- ✓ Coverage ≥85%

**Outputs:**
- artifacts/code.py

<!-- GATE_REVIEW_START --> ← Human approval required
#### Gate Review Checklist
##### Code Quality
- [ ] Code follows standards
- [ ] Security review complete
...
<!-- GATE_REVIEW_END -->
```

**Common Gate Checklist Categories:**

**For G0 (Plan Approval):**
```markdown
##### Architecture Review
- [ ] Work breakdown structure is complete and accurate
- [ ] File touch map identifies all affected components
- [ ] Dependencies and interfaces are clearly defined
- [ ] No circular dependencies introduced

##### Risk Assessment
- [ ] Risk analysis completed for proposed changes
- [ ] Mitigation strategies identified
- [ ] Rollback plan documented

##### Resource Planning
- [ ] Effort estimates are reasonable
- [ ] Timeline is achievable
- [ ] Required skills/tools identified
```

**For G1 (Code Readiness/Design Review):**
```markdown
##### Design Quality
- [ ] API contracts are well-defined
- [ ] Database schema changes are validated
- [ ] Security considerations are documented
- [ ] Performance implications are assessed

##### Architecture Compliance
- [ ] Design follows established patterns
- [ ] No architectural violations introduced
- [ ] Scalability requirements addressed
- [ ] Integration points are documented
```

**For G2 (Code Review/Implementation):**
```markdown
##### Code Quality
- [ ] Code follows project style guide
- [ ] No linting errors (ruff check passes)
- [ ] Type hints are complete (mypy passes)
- [ ] No hardcoded secrets or credentials

##### Testing
- [ ] Unit tests written for new code
- [ ] Test suite passes (pytest -q)
- [ ] Edge cases are covered
- [ ] Test coverage meets minimum threshold

##### Documentation
- [ ] Code is well-commented
- [ ] API changes are documented
- [ ] Breaking changes are highlighted
- [ ] README updated if needed
```

**For G3 (Pre-Release/Testing):**
```markdown
##### Test Coverage
- [ ] Full test suite passes
- [ ] Test coverage ≥ 85%
- [ ] No flaky tests identified
- [ ] Performance tests pass

##### Integration Testing
- [ ] End-to-end workflows tested
- [ ] Integration points validated
- [ ] Error handling verified
- [ ] Rollback tested

##### Quality Metrics
- [ ] Defect density ≤ 1.5
- [ ] No critical bugs
- [ ] No high-severity security issues
- [ ] Code review feedback addressed
```

**For G4 (Deployment Approval/Governance):**
```markdown
##### Compliance
- [ ] Decision log complete and accurate
- [ ] Compliance checklist validated
- [ ] Regulatory requirements met
- [ ] Audit trail documented

##### Deployment Readiness
- [ ] Deployment plan reviewed
- [ ] Rollback procedure validated
- [ ] Monitoring and alerting configured
- [ ] Runbook complete

##### Stakeholder Approval
- [ ] Technical lead approval obtained
- [ ] Product owner approval obtained
- [ ] Security review complete (if needed)
- [ ] Final signoff received
```

**How It Works During `spec run`:**
1. When a step with a gate review is reached
2. The checklist is displayed interactively with categories
3. Reviewer checks off completed items using questionary checkboxes
4. Reviewer makes approval decision (Approve/Reject/Defer/Conditional)
5. Decision is logged to audit trail (`.aip_artifacts/{AIP_ID}/gate_approvals.jsonl`)
6. Execution continues, pauses, or stops based on decision

**Tier-Specific Behavior:**
- **Tier A/B**: Gates are blocking - execution halts until approved
- **Tier C**: Gates are auto-approved with logging only

**View Gate Approvals:**
```bash
spec gate-list          # List all approvals
spec gate-report        # Summary statistics
```

**Example Complete Step with Gate Review:**
```markdown
### Step 1: Planning [G0: Plan Approval]

**Prompt:**

Produce detailed work breakdown and file-touch map:
- WBS with task breakdown
- Files to be modified
- Test coverage plan

**Outputs:**

- `artifacts/plan/wbs.md`
- `artifacts/plan/file-touch-map.yaml`

<!-- GATE_REVIEW_START -->
#### Gate Review Checklist

##### Architecture Review
- [ ] Work breakdown structure is complete and accurate
- [ ] File touch map identifies all affected components
- [ ] Dependencies and interfaces are clearly defined
- [ ] No circular dependencies introduced

##### Risk Assessment
- [ ] Risk analysis completed for proposed changes
- [ ] Mitigation strategies identified
- [ ] Rollback plan documented

##### Resource Planning
- [ ] Effort estimates are reasonable
- [ ] Timeline is achievable
- [ ] Required skills/tools identified

#### Approval Decision
- [ ] APPROVED
- [ ] APPROVED WITH CONDITIONS: ___
- [ ] REJECTED: ___
- [ ] DEFERRED: ___

**Approval Metadata:**
- Reviewer: ___
- Date: ___
- Rationale: ___
<!-- GATE_REVIEW_END -->
```

## Acceptance Criteria

Write specific, measurable criteria:

**Good:**
```yaml
acceptance_criteria:
  - Users can upload PNG/JPG files up to 5MB
  - Avatars display in profile page within 100ms
  - 95% test coverage on new avatar code
  - GDPR-compliant deletion within 24 hours
  - Zero accessibility violations (WCAG 2.1 AA)
```

**Bad:**
```yaml
acceptance_criteria:
  - Avatars work
  - It's fast
  - Tests pass
```

## Common Patterns

### Feature Addition (Tier C)
1. Planning Phase - outline approach
2. Implementation - write the code
3. Testing & Validation - verify it works
4. Integration - deploy to staging

### API Changes (Tier B)
1. Design Phase - API specification
2. Backend Implementation - endpoints & logic
3. Frontend Integration - consume the API
4. Testing Phase - integration tests
5. Documentation - API docs & changelog
6. Deployment - staged rollout

### System Migration (Tier A)
1. Discovery Phase - audit current system
2. Design Phase - migration strategy
3. Preparation - dual-write setup
4. Migration - data transfer
5. Validation - verify data integrity
6. Cutover - switch to new system
7. Monitoring - watch for issues
8. Cleanup - remove old system
9. Post-Mortem - lessons learned

## Tips for LLMs

1. **Be Specific:** Vague prompts lead to vague implementations. Specify file names, function signatures, test requirements.

2. **Break It Down:** Complex tasks should be multiple steps, not one giant step. Each step should take 15-30 minutes.

3. **Test Early:** Include testing in every implementation step, not just at the end.

4. **Document Decisions:** Important architectural choices should be documented in the prompt or outputs.

5. **Think Rollback:** For Tier A/B, always include rollback procedures. What if this breaks?

6. **Use Gates Wisely:** Gates slow down progress but improve quality. Place them before irreversible actions (database migrations, deployments).

7. **Reference Standards:** If your org has coding standards, API guidelines, or architecture docs, reference them in prompts.

8. **Be Explicit About "Done":** Each step should produce tangible outputs that prove it's complete.

## Validation

Before submitting your spec, check:

- [ ] All required frontmatter fields present (version, tier, title, owner, goal, orchestrator_contract, repo.working_branch)
- [ ] Acceptance criteria are specific and measurable
- [ ] Each step has a clear prompt
- [ ] Steps produce concrete outputs
- [ ] Appropriate gates for the risk tier
- [ ] Commands are valid for your environment
- [ ] Constraints are realistic and enforceable
- [ ] Background explains why this work matters now

Run `spec validate <your-spec>.md` to check against the schema.

## Example Workflow

```bash
# Initialize Specwright
spec init

# Set your default user
spec config user alice

# Create a new spec
spec create --tier C --title "Add User Avatars" --goal "Allow users to upload profile pictures"

# Edit the spec in your editor
vim .specwright/specs/add-user-avatars.md

# Validate it
spec validate .specwright/specs/add-user-avatars.md

# Compile to YAML (for orchestrators)
spec compile .specwright/specs/add-user-avatars.md

# Execute the plan (with human gates)
spec run .specwright/aips/add-user-avatars.yaml
```

## Need Help?

- Check the templates: `ls src/spec/templates/`
- View your config: `spec config --show`
- Validate before compiling: `spec validate <file>`
- Read the schema: `src/spec/schemas/aip.schema.json`

---

**Remember:** AIPs are living documents. It's okay to revise steps as you learn more during implementation. The goal is clear communication between humans and agents, not perfect foresight.
