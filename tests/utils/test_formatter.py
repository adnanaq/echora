#!/usr/bin/env python3
"""
Beautiful test result formatting using Rich library.
"""

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TestResultFormatter:
    """Beautiful formatter for test results using Rich."""

    def __init__(self):
        self.console = Console()

    def print_header(self, title: str, subtitle: str = ""):
        """Print a beautiful header for the test section."""
        header_text = Text(title, style="bold bright_cyan")
        if subtitle:
            subtitle_text = Text(subtitle, style="dim cyan")
            header_text.append("\n")
            header_text.append(subtitle_text)

        panel = Panel(header_text, box=box.DOUBLE, padding=(1, 2), style="bright_cyan")
        self.console.print(panel)

    def print_test_summary(
        self,
        test_name: str,
        total_anime: int,
        test_count: int,
        field_combinations: int,
        random_seed: int,
    ):
        """Print test configuration summary."""
        table = Table(show_header=False, box=box.ROUNDED, style="dim")
        table.add_row("📊 Total anime in database:", f"[bold]{total_anime}[/bold]")
        table.add_row("🎯 Testing anime count:", f"[bold]{test_count}[/bold]")
        table.add_row("🔢 Field combinations:", f"[bold]{field_combinations}[/bold]")
        table.add_row("🎲 Random seed:", f"[bold]{random_seed}[/bold]")

        self.console.print(
            Panel(
                table,
                title=f"[bold]{test_name} Configuration[/bold]",
                border_style="blue",
            )
        )

    def create_anime_test_panels(self) -> None:
        """Initialize for stacked information panels instead of tables."""
        # We'll use individual panels for each test instead of a table

    def print_detailed_test_result(
        self,
        test_num: int,
        anime_data: dict[str, Any],
        field_combo: list[str],
        query_text: str,
        results: list[dict[str, Any]],
        test_passed: bool,
        synonym_match: bool = False,
    ):
        """Print detailed test result using stacked information panels."""

        anime_title = anime_data.get("title", "Unknown")

        # Create the header separator
        header = f"Test #{test_num} " + "═" * (75 - len(f"Test #{test_num} "))
        self.console.print(f"\n[bold bright_blue]{header}[/bold bright_blue]")

        # Anime Title Section
        self.console.print("\n[bold bright_cyan]📽️ Anime Title[/bold bright_cyan]")
        self.console.print(f"   {anime_title}")

        # Available Fields Panel (matching multimodal style)
        field_panel_content = []
        title_fields = [
            "title",
            "title_english",
            "title_japanese",
            "synonyms",
            "synopsis",
            "background",
        ]

        for field in title_fields:
            value = anime_data.get(field)
            if value:
                if field == "synonyms":
                    if isinstance(value, list) and len(value) > 0:
                        synonyms_preview = ", ".join(value[:3])
                        if len(value) > 3:
                            synonyms_preview += f"... ({len(value)} total)"
                        field_panel_content.append(
                            f"[cyan]{field}:[/cyan] {synonyms_preview}"
                        )
                elif field in ["synopsis", "background"]:
                    text_value = str(value)
                    char_count = len(text_value)
                    preview = text_value[:100] + ("..." if char_count > 100 else "")
                    field_panel_content.append(
                        f"[cyan]{field}:[/cyan] {preview} [dim]({char_count} characters)[/dim]"
                    )
                else:
                    field_value = str(value)
                    char_count = len(field_value)
                    preview = field_value[:80] + ("..." if char_count > 80 else "")
                    field_panel_content.append(
                        f"[cyan]{field}:[/cyan] {preview} [dim]({char_count} characters)[/dim]"
                    )

        field_panel = Panel(
            "\n".join(field_panel_content),
            title="📋 Available Fields",
            border_style="blue",
            padding=(0, 1),
        )
        self.console.print(field_panel)

        # Selected Field Combination Panel (matching multimodal style)
        combo_display = " + ".join(field_combo)
        combo_panel = Panel(
            f"[yellow]{combo_display}[/yellow]",
            title="📋 Field Combinations",
            border_style="magenta",
            padding=(0, 1),
        )
        self.console.print(combo_panel)

        # Generated Query Panel (matching multimodal style)
        # Format query for better readability
        if len(query_text) > 100:
            # Split at word boundaries for readability
            words = query_text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) > 80:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = current_line + " " + word if current_line else word
            if current_line:
                lines.append(current_line)
            query_display = "\n".join(lines)
        else:
            query_display = query_text

        query_panel = Panel(
            f"[cyan]{query_display}[/cyan]",
            title="📝 Generated Text Query",
            border_style="cyan",
            padding=(0, 1),
        )
        self.console.print(query_panel)

        # Search Results Section
        self.console.print("\n[bold green]📊 Search Results[/bold green]")

        # Create a simple results table
        results_table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
        results_table.add_column("#", width=3, justify="right")
        results_table.add_column("Title", style="cyan")
        results_table.add_column("Score", width=10, justify="right")
        results_table.add_column("Status", width=8, justify="center")

        for i, result in enumerate(results[:5], 1):
            result_title = result.get("title", "Unknown")
            result_score = result.get("score", 0.0)

            # Score styling
            score_style = (
                "bright_green"
                if result_score >= 0.8
                else "yellow"
                if result_score >= 0.6
                else "white"
            )
            score_display = f"[{score_style}]{result_score:.4f}[/{score_style}]"

            # Status for first result
            status_display = ""
            if i == 1:
                if test_passed:
                    if synonym_match:
                        status_display = "[yellow]✅ SYN[/yellow]"
                    else:
                        status_display = "[green]✅ PASS[/green]"
                else:
                    status_display = "[red]❌ FAIL[/red]"

            results_table.add_row(str(i), result_title, score_display, status_display)

        self.console.print(results_table)

        # Analysis Section
        if results:
            results[0].get("score", 0.0)
            analysis_style = "bright_green" if test_passed else "red"

            if test_passed:
                if synonym_match:
                    analysis_text = "SYNONYM MATCH - Found via synonym matching!"
                elif results[0].get("title") == anime_title:
                    analysis_text = "EXACT MATCH - Perfect semantic similarity!"
                else:
                    analysis_text = "TOP-3 MATCH - Found in top results!"
            else:
                analysis_text = "NO MATCH - Target anime not found in results."

            self.console.print(
                f"\n[bold {analysis_style}]🎯 Analysis[/bold {analysis_style}]: {analysis_text}"
            )

        # Bottom separator
        separator = "═" * 75
        self.console.print(f"[dim]{separator}[/dim]")

    def print_detailed_image_test_result(
        self,
        test_num: int,
        anime_data: dict[str, Any],
        image_url: str,
        available_images: int,
        embedding_dim: int,
        results: list[dict[str, Any]],
        test_passed: bool,
    ):
        """Print detailed image test result using stacked information panels."""

        anime_title = anime_data.get("title", "Unknown")

        # Create the header separator
        header = f"Image Test #{test_num} " + "═" * (
            75 - len(f"Image Test #{test_num} ")
        )
        self.console.print(f"\n[bold bright_magenta]{header}[/bold bright_magenta]")

        # Target Anime Panel
        self.console.print("\n[bold green]🖼️ Target Anime[/bold green]")
        self.console.print(f"   {anime_title}")

        # Image Details Panel (matching multimodal style)
        image_panel_content = []
        image_panel_content.append(
            f"[bold blue]Available images:[/bold blue] {available_images}"
        )
        image_panel_content.append(f"[blue]Selected image:[/blue] {image_url}")
        image_panel_content.append(
            f"[bold green]Embedding dimension:[/bold green] {embedding_dim}D (OpenCLIP ViT-L/14)"
        )

        # Show image types if available
        if "images" in anime_data:
            images = anime_data["images"]
            image_types = []
            for img_type in ["covers", "posters", "banners", "screenshots"]:
                if img_type in images and images[img_type]:
                    count = len(images[img_type])
                    image_types.append(f"{img_type}: {count}")
            if image_types:
                image_panel_content.append(
                    f"[cyan]Image types:[/cyan] {' | '.join(image_types)}"
                )

        image_panel = Panel(
            "\n".join(image_panel_content),
            title="📸 Image Selection",
            border_style="green",
            padding=(0, 1),
        )
        self.console.print(image_panel)

        # Search Results Section
        self.console.print("\n[bold green]📊 Visual Similarity Results[/bold green]")

        # Create results table
        results_table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
        results_table.add_column("#", width=3, justify="right")
        results_table.add_column("Matched Anime", style="cyan")
        results_table.add_column("Similarity", width=12, justify="right")
        results_table.add_column("Status", width=8, justify="center")

        for i, result in enumerate(results[:5], 1):
            result_title = result.get("title", "Unknown")
            result_score = result.get("score", 0.0)

            # Score styling for visual similarity
            score_style = (
                "bright_green"
                if result_score >= 0.9
                else (
                    "yellow"
                    if result_score >= 0.7
                    else "red"
                    if result_score >= 0.5
                    else "dim"
                )
            )
            score_display = f"[{score_style}]{result_score:.4f}[/{score_style}]"

            # Status for first result
            status_display = ""
            if i == 1:
                if test_passed:
                    status_display = "[green]✅ MATCH[/green]"
                else:
                    status_display = "[red]❌ FAIL[/red]"

            results_table.add_row(str(i), result_title, score_display, status_display)

        self.console.print(results_table)

        # Analysis Section
        if results:
            top_score = results[0].get("score", 0.0)
            analysis_style = "bright_green" if test_passed else "red"

            if test_passed:
                if top_score >= 0.95:
                    analysis_text = (
                        "EXCELLENT VISUAL MATCH - Nearly perfect image similarity!"
                    )
                elif top_score >= 0.8:
                    analysis_text = (
                        "STRONG VISUAL MATCH - High image similarity detected!"
                    )
                else:
                    analysis_text = (
                        "VISUAL MATCH - Image correctly identified source anime!"
                    )
            else:
                analysis_text = "NO VISUAL MATCH - Image did not match source anime."

            self.console.print(
                f"\n[bold {analysis_style}]🎯 Visual Analysis[/bold {analysis_style}]: {analysis_text}"
            )

            # Additional insights for image similarity
            if results and len(results) >= 2:
                top_score = results[0].get("score", 0.0)
                second_score = results[1].get("score", 0.0)
                confidence_gap = top_score - second_score

                if confidence_gap > 0.3:
                    confidence_text = "Very high confidence - clear visual distinction"
                elif confidence_gap > 0.1:
                    confidence_text = "Good confidence - reasonable visual distinction"
                else:
                    confidence_text = (
                        "Lower confidence - similar visual features detected"
                    )

                self.console.print(
                    f"   [dim]Confidence gap: {confidence_gap:.4f} - {confidence_text}[/dim]"
                )

        # Bottom separator
        separator = "═" * 75
        self.console.print(f"[dim]{separator}[/dim]")

    def print_field_combination_analysis(self, field_stats: dict[str, dict[str, Any]]):
        """Print field combination effectiveness analysis."""
        if not field_stats:
            return

        table = Table(
            title="[bold]Field Combination Analysis[/bold]",
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE_HEAVY,
            expand=False,
        )

        table.add_column("Combination", style="cyan", max_width=20)
        table.add_column("T", width=3, justify="right")  # Tests
        table.add_column("P", width=3, justify="right")  # Passed
        table.add_column("Success", width=8, justify="right")
        table.add_column("Score", width=6, justify="right")
        table.add_column("Fusion", width=7, justify="right")

        # Sort by success rate
        sorted_stats = sorted(
            field_stats.items(),
            key=lambda x: x[1]["passes"] / x[1]["tests"] if x[1]["tests"] > 0 else 0,
            reverse=True,
        )

        for combo_key, stats in sorted_stats:
            if stats["tests"] > 0:
                success_rate = (stats["passes"] / stats["tests"]) * 100
                avg_score = stats["avg_score"]

                # Truncate combo key for compact display
                combo_display = (
                    combo_key[:18] + "..." if len(combo_key) > 18 else combo_key
                )

                # Success rate styling - more compact
                rate_style = (
                    "bright_green"
                    if success_rate >= 80
                    else "yellow"
                    if success_rate >= 60
                    else "red"
                )
                rate_display = f"[{rate_style}]{success_rate:.0f}%[/{rate_style}]"

                # Score styling - more compact
                score_style = (
                    "bright_green"
                    if avg_score >= 0.7
                    else "yellow"
                    if avg_score >= 0.4
                    else "red"
                )
                score_display = f"[{score_style}]{avg_score:.3f}[/{score_style}]"

                # Fusion boost if available - more compact
                fusion_display = ""
                if "fusion_boost" in stats:
                    fusion_rate = (
                        (stats["fusion_boost"] / stats["tests"]) * 100
                        if stats["tests"] > 0
                        else 0
                    )
                    fusion_style = (
                        "bright_green"
                        if fusion_rate >= 70
                        else "yellow"
                        if fusion_rate >= 40
                        else "dim"
                    )
                    fusion_display = (
                        f"[{fusion_style}]{fusion_rate:.0f}%[/{fusion_style}]"
                    )

                table.add_row(
                    combo_display,
                    str(stats["tests"]),
                    str(stats["passes"]),
                    rate_display,
                    score_display,
                    fusion_display,
                )

        self.console.print(table)

    def print_final_results(
        self,
        test_name: str,
        passed_tests: int,
        total_tests: int,
        random_seed: int,
        insights: list[str] = None,
    ):
        """Print beautiful final results summary."""
        if total_tests == 0:
            self.console.print(
                Panel(
                    "[red]❌ No tests completed[/red]",
                    title="[bold red]Test Results[/bold red]",
                    border_style="red",
                )
            )
            return

        success_rate = (passed_tests / total_tests) * 100

        # Results table
        results_table = Table(show_header=False, box=box.ROUNDED)
        results_table.add_row("🎲 Random seed:", f"[bold]{random_seed}[/bold]")
        results_table.add_row(
            "✅ Tests passed:", f"[bold green]{passed_tests}/{total_tests}[/bold green]"
        )

        # Success rate with styling
        if success_rate >= 80:
            rate_display = (
                f"[bold bright_green]{success_rate:.1f}%[/bold bright_green] 🎉"
            )
            status = "[bold bright_green]EXCELLENT[/bold bright_green]"
        elif success_rate >= 60:
            rate_display = f"[bold yellow]{success_rate:.1f}%[/bold yellow] ✅"
            status = "[bold yellow]ADEQUATE[/bold yellow]"
        else:
            rate_display = f"[bold red]{success_rate:.1f}%[/bold red] ⚠️"
            status = "[bold red]NEEDS IMPROVEMENT[/bold red]"

        results_table.add_row("📊 Success rate:", rate_display)
        results_table.add_row("🔬 Assessment:", status)

        # Create panel with appropriate border color
        border_style = (
            "bright_green"
            if success_rate >= 80
            else "yellow"
            if success_rate >= 60
            else "red"
        )

        self.console.print(
            Panel(
                results_table,
                title=f"[bold]{test_name} Results[/bold]",
                border_style=border_style,
            )
        )

        # Print insights if provided
        if insights:
            insights_text = "\n".join([f"• {insight}" for insight in insights])
            self.console.print(
                Panel(
                    insights_text,
                    title="[bold cyan]🎯 Key Insights[/bold cyan]",
                    border_style="cyan",
                )
            )

    def print_image_test_details(
        self,
        test_num: int,
        anime_title: str,
        image_count: int,
        image_url: str,
        embedding_dim: int,
    ):
        """Print image test details in a compact format."""
        details = f"🖼️ Test {test_num}: [bold cyan]{anime_title}[/bold cyan]\n"
        details += f"   📊 Available images: [yellow]{image_count}[/yellow] | "
        details += f"Generated: [green]{embedding_dim}D embedding[/green]"
        self.console.print(details)

    def print_multimodal_test_details(
        self,
        test_num: int,
        anime_title: str,
        field_combo: str,
        text_query: str,
        text_score: float,
        image_score: float,
        multimodal_score: float,
        fusion_boost: bool,
    ):
        """Print multimodal test details with score analysis."""
        details = (
            f"🎬 Multimodal Test {test_num}: [bold cyan]{anime_title}[/bold cyan]\n"
        )
        details += f"   🎲 Field combo: [yellow]{field_combo}[/yellow]\n"
        details += f"   📝 Query: [dim]{text_query[:60]}{'...' if len(text_query) > 60 else ''}[/dim]\n"

        # Score analysis with colors
        text_style = (
            "bright_green"
            if text_score >= 0.7
            else "yellow"
            if text_score >= 0.4
            else "red"
        )
        image_style = (
            "bright_green"
            if image_score >= 0.7
            else "yellow"
            if image_score >= 0.4
            else "red"
        )
        multi_style = (
            "bright_green"
            if multimodal_score >= 0.7
            else "yellow"
            if multimodal_score >= 0.4
            else "red"
        )

        details += (
            f"   📈 Scores: Text=[{text_style}]{text_score:.4f}[/{text_style}] | "
        )
        details += f"Image=[{image_style}]{image_score:.4f}[/{image_style}] | "
        details += f"Multimodal=[{multi_style}]{multimodal_score:.4f}[/{multi_style}]"

        if fusion_boost:
            details += " [bold bright_green]🎯 FUSION BOOST![/bold bright_green]"

        self.console.print(details)

    def print_final_summary(
        self,
        title: str,
        passed: int,
        total: int,
        success_rate: float,
        status_msg: str,
        status_color: str,
        random_seed: int,
    ):
        """Print final test results summary."""
        self.console.print(f"\n[bold]{title}[/bold]")

        summary_panel = Panel(
            f"[bold green]✅ Passed:[/bold green] {passed}/{total}\n"
            f"[bold blue]📈 Success Rate:[/bold blue] {success_rate:.1f}%\n"
            f"[bold yellow]🎲 Random Seed:[/bold yellow] {random_seed}\n\n"
            f"[{status_color}]{status_msg}[/{status_color}]",
            title="📊 Test Results",
            border_style=(
                "green"
                if success_rate >= 80
                else "yellow"
                if success_rate >= 60
                else "red"
            ),
            padding=(1, 2),
        )
        self.console.print(summary_panel)

    def print_error_summary(self, message: str):
        """Print error summary with Rich formatting."""
        error_panel = Panel(
            f"[bold red]{message}[/bold red]",
            title="❌ Error",
            border_style="red",
            padding=(1, 2),
        )
        self.console.print(error_panel)

    def print_detailed_multimodal_test_result(
        self,
        test_num: int,
        anime_data: dict[str, Any],
        selected_combination: list[str],
        text_query: str,
        image_url: str,
        available_images: int,
        text_score: float,
        image_score: float,
        multimodal_score: float,
        multimodal_results: list[dict[str, Any]],
        test_passed: bool,
        fusion_boost: bool,
    ):
        """Print detailed multimodal test result using stacked information panels."""

        anime_title = anime_data.get("title", "Unknown")
        combination_key = "+".join(selected_combination)

        # Create the header separator
        header = f"Multimodal Test #{test_num} " + "═" * (
            75 - len(f"Multimodal Test #{test_num} ")
        )
        self.console.print(f"\n[bold]{header}[/bold]")

        # Target Anime Panel
        self.console.print("\n[bold green]🎬 Target Anime[/bold green]")
        self.console.print(f"   {anime_title}")

        # Field Combination Panel
        field_panel_content = []
        field_panel_content.append(
            f"[bold yellow]Selected combination:[/bold yellow] {combination_key}"
        )
        field_panel_content.append(
            f"[bold blue]Available images:[/bold blue] {available_images}"
        )

        # Show field details
        for field in selected_combination:
            value = anime_data.get(field)
            if value:
                if field == "synonyms":
                    field_panel_content.append(f"   {field}: {len(value)} items")
                elif field in ["synopsis", "background"]:
                    field_panel_content.append(
                        f"   {field}: {len(str(value))} characters"
                    )
                else:
                    display_value = (
                        str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    )
                    field_panel_content.append(f"   {field}: {display_value}")

        field_panel = Panel(
            "\n".join(field_panel_content),
            title="📋 Field Combinations",
            border_style="blue",
            padding=(0, 1),
        )
        self.console.print(field_panel)

        # Query Generation Panel
        query_display = (
            text_query[:100] + "..." if len(text_query) > 100 else text_query
        )
        query_panel = Panel(
            f"[cyan]{query_display}[/cyan]",
            title="📝 Generated Text Query",
            border_style="cyan",
            padding=(0, 1),
        )
        self.console.print(query_panel)

        # Image Details Panel
        image_panel_content = f"[blue]Selected image:[/blue] {image_url}\n"
        image_panel_content += "[green]Download status:[/green] ✅ Success"

        image_panel = Panel(
            image_panel_content,
            title="📸 Image Selection",
            border_style="green",
            padding=(0, 1),
        )
        self.console.print(image_panel)

        # Score Analysis Panel
        # Determine colors based on score values
        text_style = (
            "bright_green"
            if text_score >= 0.7
            else "yellow"
            if text_score >= 0.4
            else "red"
        )
        image_style = (
            "bright_green"
            if image_score >= 0.7
            else "yellow"
            if image_score >= 0.4
            else "red"
        )
        multi_style = (
            "bright_green"
            if multimodal_score >= 0.7
            else "yellow"
            if multimodal_score >= 0.4
            else "red"
        )

        score_content = f"[{text_style}]Text-only ({combination_key}):[/{text_style}] {text_score:.4f}\n"
        score_content += (
            f"[{image_style}]Image-only:[/{image_style}] {image_score:.4f}\n"
        )
        score_content += (
            f"[{multi_style}]Multimodal (RRF):[/{multi_style}] {multimodal_score:.4f}"
        )

        score_panel = Panel(
            score_content,
            title="📈 Score Comparison",
            border_style="magenta",
            padding=(0, 1),
        )
        self.console.print(score_panel)

        # Multimodal Search Results Table
        if multimodal_results:
            table = Table(
                show_header=True,
                header_style="bold magenta",
                box=box.SIMPLE_HEAVY,
                expand=True,
            )

            table.add_column("#", width=3, justify="right", style="dim")
            table.add_column("Matched Anime", min_width=30, style="cyan")
            table.add_column("Score", width=8, justify="right")
            table.add_column("Status", width=10, justify="center")

            for idx, result in enumerate(multimodal_results[:5]):
                title = result.get("title", "Unknown")
                score = result.get("score", 0.0)

                # Truncate long titles
                if len(title) > 45:
                    title = title[:42] + "..."

                # Status indicator
                if idx == 0 and test_passed:
                    status = "[bright_green]✅ MATCH[/bright_green]"
                elif idx < 3 and test_passed:
                    status = "[green]✅ TOP-3[/green]"
                else:
                    status = ""

                # Score styling
                if score >= 0.7:
                    score_style = "bright_green"
                elif score >= 0.4:
                    score_style = "yellow"
                else:
                    score_style = "red"

                table.add_row(
                    str(idx + 1),
                    title,
                    f"[{score_style}]{score:.4f}[/{score_style}]",
                    status,
                )

            self.console.print("\n[bold]📊 Multimodal Search Results[/bold]")
            self.console.print(table)

        # Fusion Analysis and Results
        analysis_style = "bright_green" if test_passed else "red"

        if test_passed:
            if anime_title == multimodal_results[0].get("title", ""):
                analysis_text = (
                    "EXACT MULTIMODAL MATCH - Perfect fusion of text and image vectors!"
                )
            else:
                analysis_text = "TOP-3 MULTIMODAL MATCH - Successful fusion ranking!"
        else:
            analysis_text = "NO MULTIMODAL MATCH - Neither exact nor top-3 match found."

        self.console.print(
            f"\n[bold {analysis_style}]🎯 Multimodal Analysis[/bold {analysis_style}]: {analysis_text}"
        )

        # Fusion effectiveness analysis
        if fusion_boost:
            fusion_text = "🎯 FUSION BOOST - Multimodal significantly outperformed individual vectors!"
            fusion_style = "bright_green"
        elif multimodal_score >= max(text_score, image_score):
            fusion_text = (
                "🎯 FUSION BENEFIT - Multimodal improved over individual results"
            )
            fusion_style = "green"
        else:
            fusion_text = "⚠️ LIMITED FUSION - Individual vectors performed better"
            fusion_style = "yellow"

        self.console.print(
            f"   [dim]Fusion effectiveness: [{fusion_style}]{fusion_text}[/{fusion_style}][/dim]"
        )

        # Bottom separator
        separator = "═" * 75
        self.console.print(f"[dim]{separator}[/dim]")

    def print_config_summary(self, config_data: dict[str, str]):
        """Print configuration summary table."""
        table = Table(show_header=False, box=box.SIMPLE, expand=False, padding=(0, 1))

        table.add_column("Setting", style="bold blue", width=25)
        table.add_column("Value", style="cyan")

        for key, value in config_data.items():
            table.add_row(f"📊 {key}:", value)

        config_panel = Panel(
            table, title="Configuration", border_style="blue", padding=(0, 1)
        )
        self.console.print(config_panel)

    def print_detailed_staff_test_result(
        self,
        test_num: int,
        anime_data: dict[str, Any],
        field_combo: list[str],
        selected_pattern: dict,
        text_query: str,
        results: list[dict[str, Any]],
        test_passed: bool,
        staff_names: list[str],
    ):
        """Print detailed staff test result using stacked information panels."""

        anime_title = anime_data.get("title", "Unknown")

        # Create the header separator
        header = f"Staff Test #{test_num} " + "═" * (
            75 - len(f"Staff Test #{test_num} ")
        )
        self.console.print(f"\n[bold bright_cyan]{header}[/bold bright_cyan]")

        # Anime Title Section
        self.console.print("\n[bold bright_cyan]🎭 Anime Title[/bold bright_cyan]")
        self.console.print(f"   {anime_title}")

        # Staff Roles Panel
        roles_display = " + ".join(
            [role.replace("_", " ").title() for role in field_combo]
        )
        roles_panel = Panel(
            f"[yellow]{roles_display}[/yellow]",
            title="🎭 Staff Roles",
            border_style="magenta",
            padding=(0, 1),
        )
        self.console.print(roles_panel)

        # Query Pattern Panel
        pattern_content = []
        pattern_content.append(
            f"[bold blue]Pattern:[/bold blue] {selected_pattern['name']}"
        )
        pattern_content.append(
            f"[blue]Description:[/blue] {selected_pattern['description']}"
        )
        if staff_names:
            staff_preview = ", ".join(staff_names[:3])
            if len(staff_names) > 3:
                staff_preview += f"... ({len(staff_names)} total)"
            pattern_content.append(f"[cyan]Staff Names:[/cyan] {staff_preview}")

        pattern_panel = Panel(
            "\n".join(pattern_content),
            title="🎲 Query Pattern",
            border_style="blue",
            padding=(0, 1),
        )
        self.console.print(pattern_panel)

        # Generated Query Panel
        # Format query for better readability
        if len(text_query) > 100:
            # Split at word boundaries for readability
            words = text_query.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) > 80:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = current_line + " " + word if current_line else word
            if current_line:
                lines.append(current_line)
            query_display = "\n".join(lines)
        else:
            query_display = text_query

        query_panel = Panel(
            f"[cyan]{query_display}[/cyan]",
            title="📝 Generated Staff Query",
            border_style="cyan",
            padding=(0, 1),
        )
        self.console.print(query_panel)

        # Search Results Section
        self.console.print("\n[bold green]📊 Staff Vector Search Results[/bold green]")

        # Create a simple results table
        results_table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
        results_table.add_column("#", width=3, justify="right")
        results_table.add_column("Title", style="cyan")
        results_table.add_column("Score", width=10, justify="right")
        results_table.add_column("Status", width=8, justify="center")

        for i, result in enumerate(results[:5], 1):
            result_title = result.get("title", "Unknown")
            result_score = result.get("score", 0.0)

            # Score styling
            score_style = (
                "bright_green"
                if result_score >= 0.8
                else "yellow"
                if result_score >= 0.6
                else "white"
            )
            score_display = f"[{score_style}]{result_score:.4f}[/{score_style}]"

            # Status for first result
            status_display = ""
            if i == 1:
                if test_passed:
                    status_display = "[green]✅ PASS[/green]"
                else:
                    status_display = "[red]❌ FAIL[/red]"

            results_table.add_row(str(i), result_title, score_display, status_display)

        self.console.print(results_table)

        # Analysis Section
        if results:
            top_score = results[0].get("score", 0.0)
            analysis_style = "bright_green" if test_passed else "red"

            if test_passed:
                if results[0].get("title") == anime_title:
                    analysis_text = "EXACT MATCH - Perfect staff semantic similarity!"
                elif any(r.get("title") == anime_title for r in results[:5]):
                    analysis_text = "TOP-5 MATCH - Found in top results!"
                else:
                    analysis_text = "STAFF MATCH - Cross-reference validation passed!"
            else:
                analysis_text = "NO MATCH - Target anime not found in top 5 results."

            self.console.print(
                f"\n[bold {analysis_style}]🎯 Analysis[/bold {analysis_style}]: {analysis_text}"
            )

            # Additional staff-specific insights
            if test_passed and top_score >= 0.9:
                self.console.print(
                    "   [dim]Excellent staff vector quality - very high confidence match[/dim]"
                )
            elif test_passed and top_score >= 0.7:
                self.console.print(
                    "   [dim]Good staff vector quality - strong semantic similarity[/dim]"
                )

        else:
            self.console.print(
                "\n[bold red]🎯 Analysis[/bold red]: No results returned from staff vector search."
            )

        # Bottom separator
        separator = "═" * 75
        self.console.print(f"[dim]{separator}[/dim]")


# Global formatter instance
formatter = TestResultFormatter()
