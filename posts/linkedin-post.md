# LinkedIn Post

I built an AI agent from scratch — no LangChain, no frameworks. Here's the one thing that surprised me most.

Most people who "use AI agents" are calling a library. I wanted to understand what the library was actually doing.

So I built a self-debugging agent from scratch using the raw Anthropic API. Give it a repo with failing tests — it finds the bug and fixes it autonomously.

The architecture is the ReAct loop (Reasoning + Acting, Yao et al. 2022): the agent alternates between THOUGHT → ACTION → OBSERVATION until it's done.

**The thing that surprised me:** Claude has no memory between API calls. None. Every call is completely stateless.

The "memory" is a Python list. You append every thought, every tool call, every observation to it. You send the entire list on every API call. The model re-reads the whole transcript from scratch and responds as if it remembers — because it just read everything it ever did.

The context window is the agent's working memory. That's it.

**The bug that taught me the most:** Claude can return multiple tool calls in a single response. The API requires a result for every single one. Miss even one — 400 error. My first version only handled the last tool call. The fix was five lines of code and a complete shift in how I thought about the agent's action space.

Every agent framework you've ever seen — LangChain, AutoGen, the Claude SDK — is some version of this loop with scaffolding around it. Build it once from scratch and you'll understand all of them.

Full technical breakdown with code walkthrough on Medium: [link]
