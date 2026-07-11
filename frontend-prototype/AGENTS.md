# Prototype Instructions

Run the local server yourself and open the preview in the in-app browser. Do not give the user server-start instructions when you can run it.

Before making substantial visual changes, use the Product Design plugin's `get-context` skill when the visual source is unclear or no longer matches the current goal. When the user gives durable prototype-specific design feedback, preferences, or decisions, record them in `AGENTS.md`.

When implementing from a selected generated mock, treat that image as the source of truth for layout, component anatomy, density, spacing, color, typography, visible content, and hierarchy.

## Comparison Workspace Decisions

- `Reading Workspace` is a real two-mode menu: single-paper reading and multi-paper comparison.
- Comparison supports 2-4 saved papers. New PDFs return to the reading workspace for analysis before they can be selected.
- Use a compact left selector and a wide, horizontally scrollable matrix. Never squeeze four paper columns into unreadably narrow cards.
- Prefix every displayed evidence ID with its paper label (`P1:E003`) and expose the saved source preview on click.
- Keep comparison chat in an on-demand right drawer so the matrix retains the main working area on 13-inch laptop viewports.
