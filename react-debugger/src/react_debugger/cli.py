"""
CLI — entry point for the react-debugger command.

Typer handles argument parsing and --help generation.
The `run` subcommand kicks off the ReAct loop.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="react-debugger",
    help="A self-debugging agent that fixes failing tests using the ReAct loop.",
    add_completion=False,
)


@app.command()
def run(
    repo_path: str = typer.Argument(..., help="Path to the repository to debug"),
    test_command: str = typer.Option("pytest", "--test-command", "-t", help="Command to run the tests"),
    max_iterations: int = typer.Option(None, "--max-iterations", "-n", help="Override max iterations from config"),
    stub: bool = typer.Option(False, "--stub", help="Use stub tools (Phase 1 demo — no real execution)"),
) -> None:
    """Run the ReAct debugging agent against a repository."""
    from .config import settings
    from .agent import ReActAgent

    # Override config with CLI flags if provided
    if max_iterations is not None:
        settings.max_iterations = max_iterations

    typer.echo(f"Starting ReAct agent on: {repo_path}")
    typer.echo(f"Test command: {test_command}")
    typer.echo(f"Model: {settings.model}  |  Max iterations: {settings.max_iterations}")
    if stub:
        typer.echo("Mode: STUB (fake observations — Phase 1 demo)")
    typer.echo("")

    agent = ReActAgent(
        repo_path=repo_path,
        test_command=test_command,
        use_stubs=stub,
    )
    result = agent.run()

    raise typer.Exit(code=0 if result.success else 1)


if __name__ == "__main__":
    app()
