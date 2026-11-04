#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from pathlib import Path


# -------------------------------------------------------------------
# Expected filename format:
#
#   error_{CG/DG}_{SpMV/MF}_{ElementType}_p_{degree}_{nodes_type}.tex
#
# Examples:
#
#   error_CG_SpMV_Hex_p_3_equidistant.tex
#   error_DG_MF_Tet_p_4_blend_and_warp.tex
# -------------------------------------------------------------------
FILENAME_RE = re.compile(
    r"^error_"
    r"(?P<discretization>CG|DG)_"
    r"(?P<implementation>SpMV|MF)_"
    r"(?P<element>.+?)_"
    r"p_"
    r"(?P<p>\d+)_"
    r"(?P<variant>.+)"
    r"\.tex$"
)


ROW_RE = re.compile(
    r"""
    ^\s*
    (?P<cycle>\d+)
    \s*&\s*
    (?P<cells>[0-9]+)
    \s*&\s*
    (?P<dofs>[0-9]+)
    \s*&\s*
    (?P<error>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    """,
    re.VERBOSE,
)


def latex_escape(s: str) -> str:
    """
    Escape a string for use in LaTeX text, titles, captions, and legends.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    return "".join(replacements.get(ch, ch) for ch in s)


def pretty_element_name(element):
    """
    Write element names out in full.
    """
    names = {
        "Hex": "Hexahedron",
        "Tet": "Tetrahedron",
        "Quad": "Quadrilateral",
        "Tri": "Triangle",
        "Prism": "Prism",
        "Pyramid": "Pyramid",
    }

    return names.get(element, element)


def pretty_discretization_name(discretization):
    """
    Write CG/DG out in full.
    """
    names = {
        "CG": "Continuous Galerkin",
        "DG": "Discontinuous Galerkin",
    }

    return names.get(discretization, discretization)


def pretty_implementation_name(implementation):
    """
    Write MF/SpMV out in full.
    """
    names = {
        "SpMV": "Sparse Matrix Vector",
        "MF": "Matrix-Free",
    }

    return names.get(implementation, implementation)


def pretty_name(discretization, element, variant):
    """
    Create pretty names for the discretization and node variant.
    """
    discretization = pretty_discretization_name(discretization)

    if variant == "equidistant":
        variant = "Equidistant"

    elif variant == "blend_and_warp":
        if element == "Hex":
            variant = "Gauss Lobatto"
        else:
            variant = "Blend and Warp"

    else:
        variant = variant.replace("_", " ").title()

    return discretization, variant


def pretty_variant_name(element, variant):
    """
    Create a nice name for a node variant.
    """
    _, pretty_variant = pretty_name(
        discretization="CG",
        element=element,
        variant=variant,
    )

    return pretty_variant


def make_element_section_title(element):
    """
    Create the section title for one element type.

    All plot groups for the element are placed inside this section.
    """
    return pretty_element_name(element)


def make_group_subsection_title(group_key):
    """
    Create the subsection title for one discretization/implementation pair.

    group_key has the form:

        implementation, discretization, element
    """
    implementation, discretization, _ = group_key

    return (
        f"{pretty_discretization_name(discretization)}, "
        f"{pretty_implementation_name(implementation)}"
    )


def make_console_group_title(group_key):
    """
    Create a compact group name for terminal output.
    """
    implementation, discretization, element = group_key

    return (
        f"{pretty_element_name(element)}, "
        f"{discretization}, "
        f"{implementation}"
    )


def make_full_figure_title(group_key, other_variant):
    """
    Create the caption below a complete 2 by 2 figure.
    """
    implementation, discretization, element = group_key

    element_pretty = pretty_element_name(element)
    discretization_pretty = pretty_discretization_name(discretization)
    implementation_pretty = pretty_implementation_name(implementation)

    other_variant_pretty = pretty_variant_name(
        element=element,
        variant=other_variant,
    )

    return (
        f"{element_pretty}, "
        f"{discretization_pretty}, "
        f"{implementation_pretty}: "
        f"{other_variant_pretty} and Equidistant nodes"
    )


def parse_filename(path: Path):
    """
    Parse a convergence-table filename.
    """
    match = FILENAME_RE.match(path.name)

    if not match:
        return None

    return {
        "discretization": match.group("discretization"),
        "implementation": match.group("implementation"),
        "element": match.group("element"),
        "p": int(match.group("p")),
        "variant": match.group("variant"),
    }


def parse_table(path: Path):
    """
    Extract cycle, cells, degrees of freedom, and error values.
    """
    rows = []

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    for line in text.splitlines():
        line = line.strip()

        match = ROW_RE.match(line)

        if not match:
            continue

        cycle = int(match.group("cycle"))
        cells = int(match.group("cells"))
        dofs = int(match.group("dofs"))
        error = float(match.group("error"))

        rows.append(
            {
                "cycle": cycle,
                "cells": cells,
                "dofs": dofs,
                "dofs_root3": dofs ** (1.0 / 3.0),
                "error": error,
            }
        )

    rows.sort(key=lambda row: row["cycle"])

    return rows


def make_coordinates(rows, xkey):
    """
    Convert data rows to pgfplots coordinates.
    """
    coordinates = []

    for row in rows:
        x = row[xkey]
        y = row["error"]

        if xkey == "cycle":
            coordinates.append(f"({x},{y:.16e})")
        else:
            coordinates.append(f"({x:.16e},{y:.16e})")

    return " ".join(coordinates)


def make_optimal_coordinates(rows, xkey, p, align_index=1):
    """
    Create coordinates for an optimal convergence reference line.

    For dofs_root3:

        error ~ x^(-(p+1))

    For cycle:

        error ~ 2^(-(p+1) cycle)
    """
    if len(rows) <= align_index:
        return None

    order = p + 1

    reference_row = rows[align_index]
    x0 = reference_row[xkey]
    y0 = reference_row["error"]

    coordinates = []

    for row in rows:
        x = row[xkey]

        if xkey == "cycle":
            y = y0 * 2.0 ** (-order * (x - x0))
            coordinates.append(f"({x},{y:.16e})")

        elif xkey == "dofs_root3":
            y = y0 * (x / x0) ** (-order)
            coordinates.append(f"({x:.16e},{y:.16e})")

        else:
            return None

    return " ".join(coordinates)


def dataset_sort_key(dataset):
    """
    Sort datasets inside a plot.

    Sequence:
        1. Polynomial degree
        2. Node variant
        3. Filename
    """
    return (
        dataset["p"],
        dataset["variant"],
        dataset["path"].name,
    )


def element_sort_key(element):
    """
    Sort element sections.

    Known element types use the explicit order below. Unknown element
    types are placed afterwards in alphabetical order.
    """
    element_order = {
        "Hex": 0,
        "Tet": 1,
        "Quad": 2,
        "Tri": 3,
        "Prism": 4,
        "Pyramid": 5,
    }

    return (
        element_order.get(element, 99),
        element.lower(),
    )


def group_sort_key_within_element(group_key):
    """
    Sort groups inside one element section.

    Each group key is:

        implementation, discretization, element

    Required order inside every element section:

        CG SpMV
        CG MF
        DG SpMV
        DG MF
    """
    implementation, discretization, _ = group_key

    discretization_order = {
        "CG": 0,
        "DG": 1,
    }

    implementation_order = {
        "SpMV": 0,
        "MF": 1,
    }

    return (
        discretization_order.get(discretization, 99),
        implementation_order.get(implementation, 99),
    )


def plot_group_sort_key(group_key):
    """
    Sort all groups globally.

    Required hierarchy:

        1. Element type
        2. CG before DG
        3. SpMV before MF

    This produces, for example:

        Hex CG SpMV
        Hex CG MF
        Hex DG SpMV
        Hex DG MF
        Tet CG SpMV
        Tet CG MF
        Tet DG SpMV
        Tet DG MF
        ...
    """
    implementation, discretization, element = group_key

    return (
        element_sort_key(element),
        group_sort_key_within_element(group_key),
    )


def append_variant_axis(
    lines,
    datasets,
    group_key,
    variant,
    xkey,
    axis_mode_options,
    xlabel,
    ylabel,
    x_description_latex,
    panel_label,
    show_legend=False,
    right_column=False,
):
    """
    Append one subplot for one node variant.

    Each subplot contains one curve per polynomial degree.
    """
    _, _, element = group_key

    variant_pretty = latex_escape(
        pretty_variant_name(
            element=element,
            variant=variant,
        )
    )

    subplot_title = (
        rf"({panel_label}) {variant_pretty}: "
        rf"error vs {x_description_latex}"
    )

    variant_datasets = [
        dataset
        for dataset in datasets
        if dataset["variant"] == variant
    ]

    lines.append(r"\nextgroupplot[")

    if right_column:
        lines.append(
            r"    ylabel style={"
            r"at={(axis description cs:1.12,0.5)},"
            r"anchor=south,"
            r"rotate=0"
            r"},"
        )

    lines.append(rf"    xlabel={{{xlabel}}},")
    lines.append(rf"    ylabel={{{ylabel}}},")
    lines.append(rf"    {axis_mode_options},")
    lines.append(r"    grid=both,")
    lines.append(r"    clip=false,")
    lines.append(r"    mark size=1.6pt,")
    lines.append(r"    line width=0.85pt,")
    lines.append(r"    tick label style={font=\scriptsize},")
    lines.append(r"    label style={font=\small},")
    lines.append(r"    legend cell align={left},")
    lines.append(
        r"    legend style={"
        r"font=\scriptsize,"
        r"fill=white,"
        r"fill opacity=0.90,"
        r"text opacity=1,"
        r"draw=black,"
        r"rounded corners=2pt"
        r"},"
    )

    if show_legend:
        lines.append(
            r"    legend style={"
            r"at={(0.03,0.03)},"
            r"anchor=south west,"
            r"font=\scriptsize,"
            r"fill=white,"
            r"fill opacity=0.90,"
            r"text opacity=1,"
            r"draw=black"
            r"},"
        )

    lines.append(r"]")

    sorted_datasets = sorted(
        variant_datasets,
        key=dataset_sort_key,
    )

    for dataset in sorted_datasets:
        coordinates = make_coordinates(
            dataset["rows"],
            xkey,
        )

        label = latex_escape(dataset["label"])

        lines.append(
            rf"\addplot+[mark=*, solid] "
            rf"coordinates {{{coordinates}}};"
        )

        if show_legend:
            lines.append(rf"\addlegendentry{{{label}}}")

    for dataset in sorted_datasets:
        optimal_coordinates = make_optimal_coordinates(
            rows=dataset["rows"],
            xkey=xkey,
            p=dataset["p"],
            align_index=1,
        )

        if optimal_coordinates is None:
            continue

        lines.append(
            rf"\addplot+["
            rf"mark=none, "
            rf"dashed, "
            rf"black, "
            rf"line width=0.8pt"
            rf"] coordinates {{{optimal_coordinates}}};"
        )

    lines.append(
        rf"\node["
        rf"anchor=north, "
        rf"font=\small, "
        rf"align=center"
        rf"] at (rel axis cs:0.5,-0.20) "
        rf"{{{subplot_title}}};"
    )


def build_2_by_2_variant_plot(
    datasets,
    group_key,
    other_variant,
):
    """
    Build one 2 by 2 plot.

    First row:
        non-equidistant error vs cycle
        non-equidistant error vs cube root of dofs

    Second row:
        equidistant error vs cycle
        equidistant error vs cube root of dofs
    """
    variants = {
        dataset["variant"]
        for dataset in datasets
    }

    if "equidistant" not in variants:
        return ""

    if other_variant not in variants:
        return ""

    caption = latex_escape(
        make_full_figure_title(
            group_key=group_key,
            other_variant=other_variant,
        )
    )

    lines = []

    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{groupplot}[")
    lines.append(r"    group style={")
    lines.append(r"        group size=2 by 2,")
    lines.append(r"        horizontal sep=1.10cm,")
    lines.append(r"        vertical sep=2.00cm,")
    lines.append(r"    },")
    lines.append(r"    width=0.485\textwidth,")
    lines.append(r"    height=0.365\textwidth,")
    lines.append(r"]")

    append_variant_axis(
        lines=lines,
        datasets=datasets,
        group_key=group_key,
        variant=other_variant,
        xkey="cycle",
        axis_mode_options="ymode=log",
        xlabel="cycle",
        ylabel=r"$L^2$-error",
        x_description_latex="cycle",
        panel_label="a",
        show_legend=False,
        right_column=False,
    )

    append_variant_axis(
        lines=lines,
        datasets=datasets,
        group_key=group_key,
        variant=other_variant,
        xkey="dofs_root3",
        axis_mode_options="xmode=log, ymode=log",
        xlabel=r"$\sqrt[3]{\#\mathrm{dofs}}$",
        ylabel=r"$L^2$-error",
        x_description_latex=r"$\sqrt[3]{\#\mathrm{dofs}}$",
        panel_label="b",
        show_legend=True,
        right_column=True,
    )

    append_variant_axis(
        lines=lines,
        datasets=datasets,
        group_key=group_key,
        variant="equidistant",
        xkey="cycle",
        axis_mode_options="ymode=log",
        xlabel="cycle",
        ylabel=r"$L^2$-error",
        x_description_latex="cycle",
        panel_label="c",
        show_legend=False,
        right_column=False,
    )

    append_variant_axis(
        lines=lines,
        datasets=datasets,
        group_key=group_key,
        variant="equidistant",
        xkey="dofs_root3",
        axis_mode_options="xmode=log, ymode=log",
        xlabel=r"$\sqrt[3]{\#\mathrm{dofs}}$",
        ylabel=r"$L^2$-error",
        x_description_latex=r"$\sqrt[3]{\#\mathrm{dofs}}$",
        panel_label="d",
        show_legend=True,
        right_column=True,
    )

    lines.append(r"\end{groupplot}")
    lines.append(r"\end{tikzpicture}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\end{figure}")
    lines.append("")

    return "\n".join(lines)


def append_plot_group(tex, group_key, datasets):
    """
    Append all figures for one CG/DG and SpMV/MF group.

    The element section itself is created separately by
    build_latex_document().
    """
    subsection_title = latex_escape(
        make_group_subsection_title(group_key)
    )

    tex.append(rf"\subsection*{{{subsection_title}}}")
    tex.append("")

    variants = sorted(
        {
            dataset["variant"]
            for dataset in datasets
        }
    )

    other_variants = [
        variant
        for variant in variants
        if variant != "equidistant"
    ]

    if "equidistant" not in variants:
        tex.append(
            r"No equidistant data was found for this group, "
            r"so the 2 by 2 comparison plot was skipped."
        )
        tex.append("")
        return

    if not other_variants:
        tex.append(
            r"No non-equidistant data was found for this group, "
            r"so the 2 by 2 comparison plot was skipped."
        )
        tex.append("")
        return

    for other_variant in other_variants:
        plot = build_2_by_2_variant_plot(
            datasets=datasets,
            group_key=group_key,
            other_variant=other_variant,
        )

        if plot:
            tex.append(plot)


def build_latex_document(plot_groups):
    """
    Create the final LaTeX document.

    The document hierarchy is:

        Hexahedron
            Continuous Galerkin, Sparse Matrix Vector
            Continuous Galerkin, Matrix-Free
            Discontinuous Galerkin, Sparse Matrix Vector
            Discontinuous Galerkin, Matrix-Free

        Tetrahedron
            Continuous Galerkin, Sparse Matrix Vector
            Continuous Galerkin, Matrix-Free
            Discontinuous Galerkin, Sparse Matrix Vector
            Discontinuous Galerkin, Matrix-Free

        ...

    Therefore, all figures belonging to one element type are consecutive.
    """
    tex = []

    tex.append(r"\documentclass{article}")
    tex.append(r"\usepackage[a4paper,margin=1.5cm]{geometry}")
    tex.append(r"\usepackage{float}")
    tex.append(r"\usepackage{xcolor}")
    tex.append(r"\usepackage{pgfplots}")
    tex.append(r"\usepgfplotslibrary{groupplots}")
    tex.append(r"\pgfplotsset{compat=1.18}")
    tex.append(r"\usepackage{amsmath}")
    tex.append(r"\usepackage{caption}")
    tex.append(r"\captionsetup{font=small,labelfont=bf}")
    tex.append("")
    tex.append(r"\begin{document}")
    tex.append("")
    tex.append(r"\section*{Convergence plots}")
    tex.append("")
    tex.append(
        r"Each figure compares one non-equidistant node variant with "
        r"the equidistant node variant. "
        r"The first row shows the non-equidistant node variant. "
        r"The second row shows the equidistant node variant. "
        r"The left column plots the $L^2$-error over refinement cycle. "
        r"The right column plots the $L^2$-error over "
        r"$\sqrt[3]{\#\mathrm{dofs}}$. "
        r"The dashed black lines show the expected optimal order."
    )
    tex.append("")

    # Collect the available groups according to element type.
    groups_by_element = defaultdict(list)

    for group_key in plot_groups:
        _, _, element = group_key
        groups_by_element[element].append(group_key)

    # First sort by element. This ensures that all Hex figures appear
    # together, all Tet figures appear together, and so on.
    for element in sorted(groups_by_element, key=element_sort_key):
        element_title = latex_escape(
            make_element_section_title(element)
        )

        tex.append(rf"\section*{{{element_title}}}")
        tex.append("")

        group_keys = sorted(
            groups_by_element[element],
            key=group_sort_key_within_element,
        )

        # Inside an element section, the sorting order is:
        #
        #   CG SpMV
        #   CG MF
        #   DG SpMV
        #   DG MF
        for group_key in group_keys:
            append_plot_group(
                tex=tex,
                group_key=group_key,
                datasets=plot_groups[group_key],
            )

    tex.append(r"\end{document}")
    tex.append("")

    return "\n".join(tex)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate convergence plots from files named "
            "error_{CG/DG}_{SpMV/MF}_{ElementType}_"
            "p_{degree}_{nodes_type}.tex"
        )
    )

    parser.add_argument(
        "--pattern",
        default="error_*_*_*_p_*_*.tex",
        help="Glob pattern for input files.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="convergence_plots.tex",
        help="Output LaTeX file.",
    )

    args = parser.parse_args()

    files = sorted(Path(".").glob(args.pattern))

    if not files:
        raise RuntimeError(
            f"No files matching '{args.pattern}' found."
        )

    plot_groups = defaultdict(list)

    for path in files:
        info = parse_filename(path)

        if info is None:
            print(
                f"Skipping {path.name}: "
                f"filename does not match expected format"
            )
            continue

        rows = parse_table(path)

        if not rows:
            print(
                f"Skipping {path.name}: "
                f"no data rows found"
            )
            continue

        label = f"p={info['p']}"

        dataset = {
            "path": path,
            "discretization": info["discretization"],
            "implementation": info["implementation"],
            "element": info["element"],
            "p": info["p"],
            "variant": info["variant"],
            "label": label,
            "rows": rows,
        }

        # One plot group per:
        #
        #   implementation: SpMV or MF
        #   discretization: CG or DG
        #   element:        Hex, Tet, Quad, ...
        #
        # Node variants are compared inside the same 2 by 2 figure.
        group_key = (
            info["implementation"],
            info["discretization"],
            info["element"],
        )

        plot_groups[group_key].append(dataset)

    if not plot_groups:
        raise RuntimeError("No valid datasets found.")

    tex = build_latex_document(plot_groups)

    Path(args.output).write_text(
        tex,
        encoding="utf-8",
    )

    print(f"Generated {args.output}")
    print("")
    print("Convergence plot groups:")
    print("")

    current_element = None

    # Use the same ordering in the terminal report as in the LaTeX file.
    for group_key in sorted(
        plot_groups,
        key=plot_group_sort_key,
    ):
        implementation, discretization, element = group_key

        if element != current_element:
            if current_element is not None:
                print("")

            print(f"{pretty_element_name(element)}:")
            current_element = element

        datasets = plot_groups[group_key]

        variants = sorted(
            {
                dataset["variant"]
                for dataset in datasets
            }
        )

        degrees = sorted(
            {
                dataset["p"]
                for dataset in datasets
            }
        )

        has_equidistant = "equidistant" in variants
        has_other_variant = any(
            variant != "equidistant"
            for variant in variants
        )

        if has_equidistant and has_other_variant:
            status = "2 by 2 plots enabled"
        elif not has_equidistant:
            status = "skipped, missing equidistant data"
        else:
            status = "skipped, missing non-equidistant data"

        print(
            f"  {discretization} {implementation}: "
            f"p={degrees}, "
            f"variants={variants}, "
            f"{len(datasets)} datasets, "
            f"{status}"
        )


if __name__ == "__main__":
    main()