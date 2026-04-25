//! Dev-only entry point that regenerates the auto-rendered portions
//! of the TRXViz docs tree.
//!
//! Given a docs root (typically `docs/`), emits:
//!
//! - `reference/ops/<tag>.md` — one page per [`WorkflowOp`] impl with
//!   ports, parameters, description, and (when citations are present)
//!   the non-endorsement admonition + `[@key]` citation list resolved
//!   by mkdocs-bibtex.
//! - `reference/ops/index.md` — landing page grouping ops by
//!   [`OpCategory`].
//! - `examples/<slug>.svg` + `examples/<slug>.md` — curated workflow
//!   examples rendered via the hand-rolled SVG layout engine in
//!   `svg_layout`, each with the methods-boilerplate preview that the
//!   runtime report generator would produce.
//!
//! Invoked directly:
//!
//! ```bash
//! cargo run -p trxviz-docgen -- docs
//! ```

use std::path::{Path, PathBuf};

use trxviz_core::workflow::methods::{OpCategory, all_op_doc_info, generate_methods_report};

mod examples;
mod op_pages;
mod svg_layout;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let docs_root = match args.as_slice() {
        [_, dir] => PathBuf::from(dir),
        _ => {
            eprintln!("Usage: trxviz-docgen <docs-root>");
            eprintln!("  (e.g. `cargo run -p trxviz-docgen -- docs`)");
            std::process::exit(2);
        }
    };

    generate(&docs_root)?;
    println!("Wrote generated pages under {}", docs_root.display());
    Ok(())
}

fn generate(docs_root: &Path) -> anyhow::Result<()> {
    generate_op_pages(&docs_root.join("reference").join("ops"))?;
    // `docs/` sits at the repo root, so the repo root is just the
    // parent of the docs dir. Used to resolve checked-in example
    // workflow paths (`docs/assets/workflows/*.json`).
    let repo_root = docs_root.parent().unwrap_or(docs_root);
    generate_examples(&docs_root.join("examples"), repo_root)?;
    Ok(())
}

fn generate_op_pages(out_dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let infos = all_op_doc_info();

    for info in &infos {
        let page = op_pages::render_op_page(info);
        let path = out_dir.join(format!("{}.md", info.tag));
        std::fs::write(&path, page)?;
    }

    let index = op_pages::render_index(&infos);
    std::fs::write(out_dir.join("index.md"), index)?;

    eprintln!(
        "Emitted {} op pages across {} categories.",
        infos.len(),
        infos
            .iter()
            .map(|i| i.category)
            .collect::<std::collections::HashSet<OpCategory>>()
            .len()
    );
    Ok(())
}

fn generate_examples(out_dir: &Path, repo_root: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;

    let examples = examples::all_examples(repo_root)?;
    for example in &examples {
        let svg = svg_layout::render_workflow_svg(&example.document);
        std::fs::write(out_dir.join(format!("{}.svg", example.slug)), svg)?;

        let report = generate_methods_report(&example.document);
        let page = render_example_page(example, &report.body_markdown);
        std::fs::write(out_dir.join(format!("{}.md", example.slug)), page)?;
    }

    std::fs::write(out_dir.join("index.md"), render_examples_index(&examples))?;
    eprintln!("Emitted {} example workflow pages.", examples.len());
    Ok(())
}

fn render_example_page(example: &examples::Example, methods_markdown: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!("# {}\n\n", example.title));
    s.push_str(example.intro);
    s.push_str("\n\n");
    s.push_str(&format!(
        "![{title} diagram]({slug}.svg)\n\n",
        title = example.title,
        slug = example.slug,
    ));
    if let Some(commentary) = &example.commentary {
        s.push_str(commentary);
        if !commentary.ends_with('\n') {
            s.push('\n');
        }
        s.push('\n');
    }
    s.push_str("## Methods boilerplate\n\n");
    s.push_str(
        "Rendered from the runtime methods-report generator for this workflow. \
         Citation keys resolve to footnotes via mkdocs-bibtex; running the raw \
         markdown through Pandoc + the project's `citations.bib` gives the same \
         result for an exported paper.\n\n",
    );
    if methods_markdown.trim().is_empty() {
        s.push_str(
            "_This example contains no ops with methods-boilerplate prose — \
             sources and sinks are non-citable, so the generated section is \
             empty._\n",
        );
    } else {
        s.push_str(methods_markdown);
        if !methods_markdown.ends_with('\n') {
            s.push('\n');
        }
    }
    s
}

fn render_examples_index(examples: &[examples::Example]) -> String {
    let mut s = String::new();
    s.push_str("# Example workflows\n\n");
    s.push_str(
        "Curated examples that exercise the TRXViz workflow model end-to-end. \
         Each page shows the graph as an SVG diagram and the methods-boilerplate \
         text that the runtime report generator would produce for it.\n\n",
    );
    for example in examples {
        s.push_str(&format!(
            "- [{title}]({slug}.md)\n",
            title = example.title,
            slug = example.slug,
        ));
    }
    s
}
