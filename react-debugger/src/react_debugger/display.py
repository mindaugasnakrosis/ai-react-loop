"""
DISPLAY — live terminal visualization of the ReAct loop.

The visual rhythm is the whole point: THOUGHT → ACTION → OBSERVATION, repeating.
Watching it in the terminal makes the pattern click viscerally.

Rich renders panels, colored text, and rule separators. We use it to give
each loop element its own color so the eye can parse iterations at a glance:
  - Cyan    = THOUGHT  (the agent's reasoning — what it thinks is happening)
  - Yellow  = ACTION   (the tool call — what it decided to do)
  - Dim     = OBSERVATION (the result — what it found out)
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import print as rprint

# Module-level console — shared across all display calls in a run
console = Console()


def show_iteration_header(iteration: int, max_iterations: int) -> None:
    """Print a divider before each iteration so runs are easy to scan."""
    console.print()
    console.print(Rule(
        f"[bold white]Iteration {iteration} / {max_iterations}[/bold white]",
        style="dim"
    ))


def show_thought(thought: str) -> None:
    """
    Render the agent's REASONING step.

    This is the most important output to watch — it shows the agent forming
    hypotheses. A well-reasoned thought ("the range() stops one short of the
    last index") is a sign the agent understands the problem.
    """
    if not thought.strip():
        return
    console.print(Panel(
        thought.strip(),
        title="[bold cyan]THOUGHT[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    ))


def show_action(tool_name: str, tool_input: dict) -> None:
    """
    Render the ACTION step — which tool the agent chose and its parameters.

    This lets you see whether the agent's action matches its reasoning.
    A good action follows naturally from the thought above it.
    """
    # Format the key inputs as a compact summary line
    params = ", ".join(f"{k}={repr(v)}" for k, v in tool_input.items())
    action_text = Text()
    action_text.append(tool_name, style="bold yellow")
    if params:
        action_text.append(f"({params})", style="yellow")

    console.print(Panel(
        action_text,
        title="[bold yellow]ACTION[/bold yellow]",
        border_style="yellow",
        padding=(0, 1),
    ))


def show_observation(observation: str, max_lines: int = 15) -> None:
    """
    Render the OBSERVATION — what the tool returned.

    Observations are truncated so long test outputs don't scroll the thought
    off screen. The truncation line tells you how much was cut.
    """
    lines = observation.splitlines()
    if len(lines) > max_lines:
        visible = "\n".join(lines[:max_lines])
        hidden = len(lines) - max_lines
        visible += f"\n[dim]... +{hidden} more lines[/dim]"
    else:
        visible = observation

    console.print(Panel(
        visible,
        title="[bold dim]OBSERVATION[/bold dim]",
        border_style="dim",
        padding=(0, 1),
    ))


def show_status(iteration: int, tokens_used: int, tests_passing: bool | None) -> None:
    """One-line status after each observation — tokens spent and test state."""
    status_parts = [f"tokens used: {tokens_used:,}"]
    if tests_passing is True:
        status_parts.append("[bold green]tests: PASSING[/bold green]")
    elif tests_passing is False:
        status_parts.append("[bold red]tests: FAILING[/bold red]")

    console.print("  " + "  ·  ".join(status_parts))


def show_finish(success: bool, summary: str, root_cause: str | None) -> None:
    """Final panel shown when the agent calls finish()."""
    color = "green" if success else "red"
    label = "SOLVED" if success else "GAVE UP"

    body = summary
    if root_cause:
        body += f"\n\n[bold]Root cause:[/bold] {root_cause}"

    console.print()
    console.print(Panel(
        body,
        title=f"[bold {color}]{label}[/bold {color}]",
        border_style=color,
        padding=(1, 2),
    ))


def show_max_iterations_warning(max_iterations: int) -> None:
    console.print()
    console.print(
        f"[bold red]⚠  Reached maximum iterations ({max_iterations}). "
        "Giving the agent one final turn to summarise.[/bold red]"
    )


def show_sandbox_event(message: str) -> None:
    """Status line for Docker sandbox lifecycle (start / stop)."""
    console.print(f"  [dim blue]⬡ {message}[/dim blue]")


def show_error(message: str) -> None:
    console.print(f"[bold red]ERROR:[/bold red] {message}")


def show_run_summary(
    iterations: int,
    total_tokens: int,
    elapsed_seconds: float,
    success: bool,
) -> None:
    """Final statistics line after a run completes."""
    outcome = "[bold green]SUCCESS[/bold green]" if success else "[bold red]FAILURE[/bold red]"
    console.print()
    console.print(Rule(style="dim"))
    console.print(
        f"  {outcome}  ·  {iterations} iterations  ·  "
        f"{total_tokens:,} tokens  ·  {elapsed_seconds:.1f}s"
    )
    console.print()
