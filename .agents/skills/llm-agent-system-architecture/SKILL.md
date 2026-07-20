---
name: llm-agent-system-architecture
description: Expert guidelines for designing LLM agentic systems, subagent task delegation, tool schema design, context window management, structured JSON parsing, prompt routing, and reactive async event loops.
---

# LLM & AI Agent System Architecture Skill

Use this skill when architecting AI agent systems, designing tool definitions, configuring subagent workflows, or managing context window efficiency.

## Core Agentic Design Principles

1. **Tool Schema & Parameter Precision**:
   - Define unambiguous JSON schemas for agent tools. Always provide explicit descriptions for every parameter and tool action.
   - Enforce parameter type validation (e.g. string paths, integer bounds, boolean flags) to prevent tool invocation failures.

2. **Subagent Task Delegation & Context Isolation**:
   - Delegate specialized sub-tasks (e.g., long-running builds, isolated browser automation, heavy data processing) to subagents.
   - Provide comprehensive, autonomous prompts to subagents containing all necessary context, target paths, and explicit completion reporting requirements.

3. **Reactive Async Event Loops (No-Polling Principle)**:
   - Avoid polling loops (`while status != 'DONE'`) when background processes or timers are running.
   - Utilize background task management and reactive timers (`schedule` tool) so execution resumes automatically when events fire.

4. **Structured Output & Resilient Error Recovery**:
   - Parse tool and model outputs defensively. Wrap JSON decoding and string parsing in `try-except` blocks.
   - Implement graceful fallbacks when external APIs or model invocations fail.

5. **Context Window Efficiency**:
   - Utilize progressive disclosure: keep system prompts and global rules compact.
   - Load specialized domain knowledge (`SKILL.md`) dynamically on-demand when relevant task triggers are detected.
