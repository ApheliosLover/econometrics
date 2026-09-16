#!/usr/bin/env python3
"""书页 → PDF 页码换算（李子奈《计量经济学》第五版）。

教材 PDF 前面有 29 页版权页/序言/目录，且在书 p.124 处又多出 2 页，
所以偏移量不是一个常数：

    书 p.1–123   → PDF 页 = 书页 + 29
    书 p.124–275 → PDF 页 = 书页 + 31

用法：
    python3 tools/bookpage.py 43         # 单页
    python3 tools/bookpage.py 105 148    # 范围
"""
import sys

BOOK_LAST = 275


def pdf_page(book_page: int) -> int:
    return book_page + (29 if book_page < 124 else 31)


def main(argv: list[str]) -> int:
    if not 1 <= len(argv) <= 2:
        print(__doc__)
        return 1
    try:
        pages = [int(a) for a in argv]
    except ValueError:
        print(f"页码必须是整数，收到：{' '.join(argv)}")
        return 1

    for p in pages:
        if not 1 <= p <= BOOK_LAST:
            print(f"书页 {p} 超出正文范围 1–{BOOK_LAST}")
            return 1

    if len(pages) == 1:
        print(f"书 p.{pages[0]}  →  PDF 第 {pdf_page(pages[0])} 页")
    else:
        lo, hi = sorted(pages)
        print(f"书 p.{lo}–{hi}  →  PDF 第 {pdf_page(lo)}–{pdf_page(hi)} 页"
              f"（共 {hi - lo + 1} 页）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
