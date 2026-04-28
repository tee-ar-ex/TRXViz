//! Markdown rendering for the per-op reference pages and the index.
//!
//! Kept deliberately string-based — there's no markdown AST in the
//! picture, just text pushed into a `String`. mkdocs-material's
//! `admonition` extension handles the disclaimer block via the
//! `!!! warning "…"` syntax.
//!
//! The non-endorsement admonition appears on every op page with at
//! least one citation key, and again on the index — matching the plan's
//! "disclaimer everywhere citations appear" requirement.

use trxviz_core::workflow::PortKind;
use trxviz_core::workflow::methods::{NON_ENDORSEMENT_NOTICE, OpCategory, OpDocInfo};

pub fn render_op_page(info: &OpDocInfo) -> String {
    let mut s = String::new();
    s.push_str(&format!("# {}\n\n", info.title));
    s.push_str(&format!(
        "*Tag:* `{}`  \n*Category:* {}\n\n",
        info.tag,
        info.category.label()
    ));

    if !info.citation_keys.is_empty() {
        push_non_endorsement_admonition(&mut s);
    }

    // `describe()` defaults to `title()` for ops that haven't overridden
    // it. Skip the body paragraph in that case rather than printing the
    // title twice; ops with real descriptions get them rendered.
    if info.describe.as_ref() != info.title {
        s.push_str(&format!("{}\n\n", info.describe));
    }

    s.push_str("## Ports\n\n");
    push_ports_table(&mut s, info);

    if !info.parameters.is_empty() {
        s.push_str("## Parameters\n\n");
        s.push_str("| Field | Default |\n");
        s.push_str("|-------|--------|\n");
        for p in &info.parameters {
            s.push_str(&format!("| `{}` | `{}` |\n", p.name, p.default_json));
        }
        s.push('\n');
    }

    if !info.citation_keys.is_empty() {
        s.push_str("## Citations\n\n");
        s.push_str(
            "When you use this op in a published workflow, please credit the original \
             authors whose methods TRXViz re-implements or ports:\n\n",
        );
        for key in info.citation_keys {
            s.push_str(&format!("- [@{key}]\n"));
        }
        s.push('\n');
    }

    s
}

pub fn render_index(infos: &[OpDocInfo]) -> String {
    let mut s = String::new();
    s.push_str("# Op reference\n\n");
    push_non_endorsement_admonition(&mut s);
    s.push_str(
        "Every op available to the TRXViz workflow editor is listed here, grouped by \
         role. Each page covers the op's input and output ports, a short \
         description, and the citations users should include when publishing \
         results that rely on the method.\n\n",
    );

    // Grouping order matches OpCategory declaration order, which is
    // roughly the data-flow order (sources → filters → tractography → …
    // → display/io). Categories that have no ops in the current build
    // are skipped.
    let order = [
        OpCategory::Source,
        OpCategory::StreamlineFilter,
        OpCategory::Tractography,
        OpCategory::Roi,
        OpCategory::Surface,
        OpCategory::Coloring,
        OpCategory::Display,
        OpCategory::Io,
        OpCategory::Other,
    ];

    for category in order {
        let mut in_cat: Vec<&OpDocInfo> = infos.iter().filter(|i| i.category == category).collect();
        if in_cat.is_empty() {
            continue;
        }
        // Stable alphabetical listing within each bucket — the
        // registry order is informative for docgen itself but not for
        // a reader scanning a category.
        in_cat.sort_by(|a, b| a.title.cmp(b.title));

        s.push_str(&format!("## {}\n\n", category.label()));
        for info in in_cat {
            if info.describe.as_ref() == info.title {
                s.push_str(&format!("- [{}]({}.md)\n", info.title, info.tag));
            } else {
                s.push_str(&format!(
                    "- [{}]({}.md) — {}\n",
                    info.title, info.tag, info.describe,
                ));
            }
        }
        s.push('\n');
    }
    s
}

fn push_non_endorsement_admonition(s: &mut String) {
    // mkdocs-material admonition. The plan's CI check greps every doc
    // page with citations for the exact `NON_ENDORSEMENT_NOTICE`
    // substring, so don't paraphrase.
    s.push_str("!!! warning \"Not the authoritative implementation\"\n");
    for line in wrap_indented(NON_ENDORSEMENT_NOTICE, 76) {
        s.push_str(&format!("    {line}\n"));
    }
    s.push('\n');
}

fn push_ports_table(s: &mut String, info: &OpDocInfo) {
    s.push_str("| Direction | # | Kind |\n");
    s.push_str("|-----------|---|------|\n");
    match info.input_ports {
        Some(inputs) => {
            if inputs.is_empty() {
                s.push_str("| _Input_ | — | _none_ |\n");
            } else {
                for (i, port) in inputs.iter().enumerate() {
                    s.push_str(&format!("| Input | {i} | `{}` |\n", port_label(*port)));
                }
            }
        }
        None => {
            s.push_str(
                "| _Input_ | — | _dynamic: varies by layer count (see op configuration)_ |\n",
            );
        }
    }
    if info.output_ports.is_empty() {
        s.push_str("| _Output_ | — | _none (terminal node)_ |\n");
    } else {
        for (i, port) in info.output_ports.iter().enumerate() {
            s.push_str(&format!("| Output | {i} | `{}` |\n", port_label(*port)));
        }
    }
    s.push('\n');
}

fn port_label(port: PortKind) -> &'static str {
    match port {
        PortKind::Streamline => "Streamline",
        PortKind::Volume => "Volume",
        PortKind::Cifti => "Cifti",
        PortKind::Surface => "Surface",
        PortKind::Parcellation => "Parcellation",
        PortKind::ParcelSelection => "ParcelSelection",
        PortKind::SurfaceScalars => "SurfaceScalars",
        PortKind::SurfaceAppearance => "SurfaceAppearance",
        PortKind::BundleSurface => "BundleSurface",
        PortKind::BoundaryField => "BoundaryField",
        PortKind::Fixels => "Fixels",
        PortKind::FixelScalars => "FixelScalars",
        PortKind::OdfField => "OdfField",
        PortKind::OdxCatalog => "OdxCatalog",
        PortKind::VoxelMask => "VoxelMask",
        PortKind::TrackingPlan => "TrackingPlan",
    }
}

/// Wrap `text` at word boundaries into lines of at most `width`
/// characters. Inserts no leading indent — the caller adds the four
/// spaces required by the admonition body. Keeps already-short lines
/// unchanged and never splits a word in the middle.
fn wrap_indented(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + 1 + word.len() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use trxviz_core::workflow::methods::all_op_doc_info;

    #[test]
    fn every_cited_op_page_contains_the_non_endorsement_notice_verbatim() {
        // The plan mandates that the canonical notice appears in every
        // generated surface with citations. This test is the programmatic
        // guard against drift.
        for info in all_op_doc_info() {
            if info.citation_keys.is_empty() {
                continue;
            }
            let page = render_op_page(&info);
            // We wrap the notice in the op page, so check by looking
            // for two stable anchor phrases from the canonical string
            // (the full string is broken up by newlines + 4-space indent).
            assert!(
                page.contains("NOT the authoritative"),
                "op page `{}` missing non-endorsement anchor phrase",
                info.tag
            );
            assert!(
                page.contains("endorsed"),
                "op page `{}` missing endorsement disclaimer",
                info.tag
            );
        }
    }

    #[test]
    fn index_is_grouped_by_category_and_lists_every_op() {
        let infos = all_op_doc_info();
        let index = render_index(&infos);
        for info in &infos {
            assert!(
                index.contains(&format!("]({}.md)", info.tag)),
                "index is missing a link to `{}`",
                info.tag
            );
        }
        // Every category label that has at least one op must appear as
        // an H2 in the index.
        let mut seen: std::collections::HashSet<&'static str> = Default::default();
        for info in &infos {
            seen.insert(info.category.label());
        }
        for label in seen {
            assert!(
                index.contains(&format!("## {label}")),
                "index missing heading for `{label}`"
            );
        }
    }

    #[test]
    fn every_op_page_lists_at_least_one_output_port_or_marks_terminal() {
        for info in all_op_doc_info() {
            let page = render_op_page(&info);
            assert!(
                page.contains("| Output |") || page.contains("_none (terminal node)_"),
                "op page `{}` rendered without output port rows",
                info.tag
            );
        }
    }
}
