# Contributing to FORGE Curated

Thank you for your interest in contributing to FORGE Curated! This document provides guidelines for different types of contributions. We welcome community involvement to help grow and improve this dataset.

---

## Table of Contents

- [Contributing to FORGE Curated](#contributing-to-forge-curated)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Ways to Contribute](#ways-to-contribute)
    - [1. Reporting Data Errors](#1-reporting-data-errors)
    - [2. Submitting New Audit Reports](#2-submitting-new-audit-reports)
    - [3. Requesting New Audit Teams](#3-requesting-new-audit-teams)
  - [Pull Request Workflow](#pull-request-workflow)
  - [Data Standards](#data-standards)
  - [Questions?](#questions)

---

## Code of Conduct

Please be respectful and constructive in all interactions. This is a research-oriented project, we value all accuracy, transparency, and collaborative improvement.

---

## Ways to Contribute

### 1. Reporting Data Errors

If you find an error in a finding's CWE classification, severity, code location, or any other field, please open a **GitHub Issue** using the following format:

**Title:** `[Fix] <AuditorName> - <FindingTitle> - <ErrorType>`

**Body:**
```
**Finding ID:** (e.g., vfp_00016 or the finding `id` field in the JSON)
**Report Path:** dataset-curated/findings/<AuditorName>/xxx.json
**Field:** (e.g., category / location / severity)
**Current Value:** ...
**Suggested Value:** ...
**Reason / Reference:** (link to report page, line number, etc.)
```

We will verify and merge fixes on a rolling basis. You are also welcome to submit a PR directly — see the [Pull Request Workflow](#pull-request-workflow) section below.

---

### 2. Submitting New Audit Reports

We welcome contributions of new audit reports from existing tracked auditors or new ones.

**Requirements:**
- The audited project's source code must be **publicly accessible** (open-source on GitHub/GitLab/Etherscan/Bscscan/Polygonscan/Basescan/...).
- The audit report must be **publicly available** (PDF or equivalent).
- The report must cover **EVM-compatible smart contracts** (Solidity).

**Steps:**

1. Fork this repository and create a new branch:
   ```bash
   git checkout -b feature/add-<AuditorName>-<ProjectName>
   ```
   or add in batch if you have multiple reports from the same auditor:

   ```bash
   git checkout -b feature/add-<AuditorName>-<DateScope>
   ```

2. Add the PDF report to the correct directory:
   ```
   dataset-curated/reports/<AuditorName>/<YYYY-MM-projectname-report>.pdf
   ```
   Follow the existing naming convention in that directory.

3. If you are also providing extracted findings JSON, place it in:
   ```
   dataset-curated/findings/<AuditorName>/<YYYY-MM-projectname-report>.json
   ```
   Ensure it conforms to the [data schema](models/schemas.py). See an [example finding](README.md#examples) in the README.

4. Open a Pull Request to the `dev` branch (not `main`). See [Pull Request Workflow](#pull-request-workflow).

> [!NOTE]
> If you are only submitting the PDF without extracted findings, that is perfectly fine. We will process the findings extraction using the FORGE framework on our end.

---

### 3. Requesting New Audit Teams

If you would like us to track a specific audit team not currently included, open a **GitHub Issue** with:

- Audit team name
- Website / official profile link
- Link to their public reports archive (e.g., GitHub repo, website reports page)
- Brief note on why you think their reports are a good fit (report quality, volume, coverage, etc.)

---

## Pull Request Workflow

All PRs should target the **`dev` branch**, not `main`. The `main` branch is reserved for stable, verified releases.

1. **Fork** the repository and clone locally.
2. Create a descriptive branch from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/add-mixbytes-2025-q1
   ```
3. Make your changes, then commit with a clear message:
   ```bash
   git commit -m "feat: add MixBytes audit reports for [ProjectName]"
   ```
   Suggested commit prefixes:
   - `feat:` — new reports or findings
   - `fix:` — corrections to existing data
   - `docs:` — documentation updates
   - `chore:` — formatting, renaming, maintenance

4. Push and open a PR on GitHub:
   ```bash
   git push origin feature/add-mixbytes-2025-q1
   ```
   In your PR description, please include:
   - What was added or changed
   - Source link(s) to the original report(s)
   - Whether extracted findings JSON is included
   - Any known limitations or caveats

5. We will review and merge into `dev`. Verified batches are periodically merged into `main` as dataset releases.

---

## Data Standards

To keep the dataset consistent, please follow these standards when contributing findings JSON:

- **Schema:** All JSON files must conform to the Pydantic models defined in [`models/schemas.py`](models/schemas.py).
- **Commit ID:** Always reference the exact commit audited, as stated in the report.
- **Severity:** Use one of: `Critical`, `High`, `Medium`, `Low`, `Informational`, or a null value (`null`).
- **CWE Classification:** Use the [CWE dictionary](models/cwe_dict.json) for reference. Assign CWEs following a tree-like structure as in existing examples.
- **Location Format:** `location` field Follow the `filename.sol::FunctionName#StartLine-EndLine` convention. The `files` field should be a list of all relevant files and can be precisely located by `Path(project_info.project_path.values[0]) / files[x]`.



---

## Questions?

Feel free to open an Issue for any questions, suggestions, or general feedback. We appreciate all forms of community engagement.