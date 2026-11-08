"""Build report.pdf from report.md (images are resolved relative to this folder).

    python report/build_pdf.py   (requires: pip install markdown-pdf)
"""
import re
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

HERE = Path(__file__).resolve().parent
WIDTH = {"task2_train_vs_val.png": 100, "task4_head_scores.png": 100,
         "task4_attn_shakespeare.png": 92, "task4_attn_modern_prose.png": 92, "task4_attn_repeated.png": 92}
CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 8.8pt; line-height: 1.22; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; }
h2 { font-size: 11.5pt; margin: 6pt 0 2pt 0; border-bottom: 1px solid #999; }
p, li { margin: 1pt 0; }
ul, ol { margin: 1pt 0; }
table { border-collapse: collapse; font-size: 7.6pt; margin: 3pt 0; }
th, td { border: 1px solid #bbb; padding: 1px 3px; }
th { background: #eee; }

hr { display: none; }
code { font-size: 8.5pt; }
"""


def main():
    # the converter only loads images inside `root`, so resolve ../figures/ from the repo root
    md = (HERE / "report.md").read_text().replace("](../figures/", "](figures/")
    def img(m):
        alt, src = m.group(1), m.group(2)
        w = WIDTH.get(Path(src).name, 62)
        return f'<img src="{src}" alt="{alt}" style="width:{w}%"/>'
    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", img, md)
    pdf = MarkdownPdf(toc_level=0)
    pdf.meta["title"] = "HW1: Language Models from Scratch (Part 1)"
    pdf.meta["author"] = "Anurag Bodapally"
    pdf.add_section(Section(md, root=str(HERE.parent), paper_size="Letter"), user_css=CSS)
    out = HERE / "report.pdf"
    tmp = HERE / "report.tmp.pdf"
    pdf.save(str(tmp))
    compress(tmp, out)
    tmp.unlink()
    print(f"[saved] {out} ({out.stat().st_size / 1e6:.1f} MB)")


def compress(src: Path, dst: Path, max_px: int = 1600):
    """markdown-pdf stores images uncompressed; re-encode them as optimized PNGs and deflate everything."""
    import pymupdf
    doc = pymupdf.open(src)
    done = set()
    for page in doc:
        for xref, *_ in page.get_images(full=True):
            if xref in done:
                continue
            done.add(xref)
            pix = pymupdf.Pixmap(doc, xref)
            if pix.alpha:
                pix = pymupdf.Pixmap(pix, 0)
            scale = max(pix.width, pix.height) / max_px
            if scale > 1:
                pix = pymupdf.Pixmap(pix, int(pix.width / scale), int(pix.height / scale), None)
            page.replace_image(xref, pixmap=pix)
    doc.save(dst, garbage=4, deflate=True, deflate_images=True, clean=True)


if __name__ == "__main__":
    main()
