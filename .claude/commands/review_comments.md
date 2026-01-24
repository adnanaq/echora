---
allowed-tools: Read, Edit, Write, Bash(git:*), Grep, Glob, Task, mcp__serena__*
description: Validate and implement technical feedback from a specific role's perspective
argument-hint: <role> <comment>
model: claude-sonnet-4-5-20250929
---

# Review Comments Command

**Role**: $ARGUMENTS[0]
**Comment**: $ARGUMENTS[1]

## Objective

You are acting as a **$ARGUMENTS[0]** reviewing technical feedback. Your job is to use relevent superpowers skill(s) to:

1. **Validate** - Assess the technical accuracy and validity of the comment/suggestion
2. **Verify** - Use your expertise as a $ARGUMENTS[0] to determine if the feedback is grounded in sound principles
3. **Act** - Either implement valid suggestions OR explain why invalid ones are incorrect

## Process

### Step 1: Analysis

Analyze the comment from the perspective of a **$ARGUMENTS[0]**:

- Is this technically sound?
- Does it align with best practices in this domain?
- Are the assumptions valid?
- What evidence supports or contradicts this?

### Step 2: Decision

Based on your analysis, choose one path:

**Path A - Valid Feedback**: If the comment is technically correct and justified:

- Acknowledge why it's valid
- Identify what needs to change
- Implement the necessary fixes/improvements
- Test/verify the changes

**Path B - Invalid Feedback**: If the comment is incorrect or based on flawed assumptions:

- Explain precisely why it's incorrect
- Provide evidence from $ARGUMENTS[0] domain knowledge
- Offer the correct understanding or approach
- DO NOT implement changes

### Step 3: Documentation

- Summarize what was validated
- Document what was implemented (if Path A)
- Explain the reasoning clearly

## Guidelines

- **Be objective**: No bias toward implementing or rejecting
- **Be precise**: Use specific technical terminology
- **Be professional**: Respectful but honest assessment
- **Be thorough**: Check assumptions, verify facts, test implementations
- **Use tools**: Search codebase, read files, check documentation as needed

## Current Context

Project: Echora (anime semantic search microservice)
Tech Stack: Pants, Python, FastAPI, Qdrant, vector embeddings
Role: **$ARGUMENTS[0]**

Review Comment:

```
$ARGUMENTS[1]
```

---

**Begin your analysis now.**
