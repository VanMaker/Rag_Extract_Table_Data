import pdfplumber
import pandas as pd
import cv2
import numpy as np
import pytesseract
import csv
from pathlib import Path
from pypdf import PdfReader, PdfWriter, PageObject
from pdf2image import convert_from_path
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import re, math, sys, pathlib
import pandas as pd
import click
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 表格处理
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """图像预处理：增强对比度、去除噪点"""
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 自适应二值化
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 降噪
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cleaned

# 视觉模块检测表格线,并生成单元格坐标
def detect_table_lines(image: np.ndarray) -> Tuple[np.ndarray, list]:
    # 检测水平线和垂直线
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    # 生成单元格坐标
    if lines is not None:
        return image, [(0, 0, image.shape[1], image.shape[0])]  # 示例返回整个图像区域
    else:
        return image, []


def ocr_cell(cell_img: np.ndarray, lang: str = 'chi_sim+eng') -> str:
    """识别单个单元格文字"""
    text = pytesseract.image_to_string(
        cell_img,
        lang=lang,
        config='--psm 6 --oem 3'  # 单行文本识别模式
    )
    return text.strip()


def process_page(args: Tuple) -> Optional[pd.DataFrame]:
    """处理单个页面"""
    pdf_path, page_num, dpi, poppler_path = args
    try:
        # PDF转图像
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
            poppler_path=poppler_path
        )
        if not images:
            return None

        # 转换OpenCV格式
        img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)

        # 预处理
        processed = preprocess_image(img)

        # 检测表格区域（简单实现）
        _, cells = detect_table_lines(processed)

        # 提取单元格数据
        data = []
        for (x1, y1, x2, y2) in cells:
            cell_img = processed[y1:y2, x1:x2]
            text = ocr_cell(cell_img)
            data.append([text])

        return pd.DataFrame(data, columns=['内容'])

    except Exception as e:
        print(f"第 {page_num} 页处理失败: {str(e)}")
        return None


# 拼接表格所跨的连续两页
def concat_page(src_pdf: str | Path,
                            page_a: int,
                            page_b: int,
                            dst_pdf: str | Path = "concat.pdf",
                            vertical: bool = True) -> None:
    """ 将表格所跨两页拼到一页，写入一个pdf`dst_pdf`.页码默认 1 开始；`vertical=True` 纵向上下拼接，反之横向拼接。 """
    reader = PdfReader(str(src_pdf))
    total_pages = len(reader.pages)

    # 简单校验
    if not (1 <= page_a <= total_pages and 1 <= page_b <= total_pages):
        raise ValueError("页码超出范围")

    # 取出 PageObject（索引需减 1）
    p1: PageObject = reader.pages[page_a - 1]
    p2: PageObject = reader.pages[page_b - 1]

    # 获取宽高（PyPDF 中尺寸以 PDF 用户单位，1 pt≈1/72 in）
    w1, h1 = float(p1.mediabox.width),  float(p1.mediabox.height)
    w2, h2 = float(p2.mediabox.width),  float(p2.mediabox.height)

    # 新页面尺寸
    if vertical:
        new_w  = max(w1, w2)
        new_h  = h1 + h2
    else:
        new_w  = w1 + w2
        new_h  = max(h1, h2)

    writer = PdfWriter()
    new_page = PageObject.create_blank_page(width=new_w, height=new_h)

    # 放置两页：PyPDF 的坐标原点在左下
    if vertical:
        new_page.merge_translated_page(p1, 0, h2)          # p1 在上
        new_page.merge_page(p2)                            # p2 在下
    else:
        new_page.merge_page(p1)                            # p1 在左
        new_page.merge_translated_page(p2, w1, 0)          # p2 在右

    writer.add_page(new_page)

    # 若源文件还有其它页可以按需 writer.add_page(reader.pages[i]) 附加
    with open(dst_pdf, "wb") as f:
        writer.write(f)


# 将上述函数的输出结果作为函数的输入，得到最终的目标数据
def get_data(RAW_FILE:str,OUT_FILE:str,KEYWORD:str) -> pd.DataFrame:
    # 读取原始文本
    raw_df = pd.read_csv(RAW_FILE)
    raw_text = "\n".join(raw_df.iloc[:, 0].astype(str).tolist())

    # 抽取列名
    header_line = next(
        line for line in raw_text.splitlines()
        if "Copper uranium gold operation" in line and "|" in line
    )
    header_segment = re.search(r"\|([^|]+?)\|", header_line).group(1).strip()
    base_headers = header_segment.split()  # ['Mt', '%Cu', 'kg/tU3O8', ...]

    # 找到目标数据行
    target_line = next(
        line.strip() for line in raw_text.splitlines()
        if KEYWORD in line
    )

    # 解析行
    tokens = target_line.split()
    num_re = re.compile(r"^[\d,]+(?:\.\d+)?$")

    # Deposit & OreType
    deposit_name = " ".join(tokens[:2])  # Olympic Dam
    ore_tokens = []
    idx = 2
    while idx < len(tokens) and not num_re.match(tokens[idx]):
        ore_tokens.append(tokens[idx]);
        idx += 1
    ore_type = " ".join(ore_tokens)

    # 数值字段
    numbers = [t.replace(",", "") for t in tokens[idx:]
               if num_re.match(t)]
    groups = math.ceil(len(numbers) / len(base_headers))
    num_headers = [
                      f"{h}_{g}" for g in range(1, groups + 1)
                      for h in base_headers
                  ][: len(numbers)]

    # 组装DataFrame
    row = [deposit_name, ore_type] + numbers
    cols = ["Deposit", "OreType"] + num_headers
    df = pd.DataFrame([row], columns=cols)
    df[num_headers] = df[num_headers].apply(pd.to_numeric, errors="coerce")

    df.to_csv(OUT_FILE, index=False)
    print(f"数据提取结果保存到{OUT_FILE!s}")


def extract_scanned_pdf(
        pdf_path: str,
        pages: List[int] = [1],
        dpi: int = 300,
        workers: int = 4,
        poppler_path: str = r"C:\poppler\Library\bin"  # poppler的本地路径
) -> pd.DataFrame:
    """提取PDF数据"""
    # 获取总页数
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

    # 确定处理页数
    if not pages:
        pages = list(range(1, total_pages + 1))
    else:
        pages = [p for p in pages if 1 <= p <= total_pages]

    # 多进程处理
    with ProcessPoolExecutor(max_workers=workers) as executor:
        args = [(pdf_path, p, dpi, poppler_path) for p in pages]
        results = list(executor.map(process_page, args))

    # 合并结果
    valid_dfs = [df for df in results if df is not None and not df.empty]
    if not valid_dfs:
        raise ValueError("所有页面处理失败，请检查：\n"
                         "1. Poppler路径是否正确\n"
                         "2. PDF文件是否加密\n"
                         "3. 图像分辨率是否过低")

    return pd.concat(valid_dfs, ignore_index=True)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--pdf-path', help='pdf file you want to process')
@click.option('--num1', type=int, default=1, help='the first page')
@click.option('--num2', type=int, default=2, help='the second page')
@click.option('--target-file', default="concat.pdf", help='target file')
@click.option('--keyword', default="Olympic Dam", help='the keyword you want to extract')
@click.option('--method', default=False, help='concatenate method:horizontal or vertical')
def final_extract(pdf_path, num1, num2, target_file, keyword, method):
    concat_page(pdf_path, num1, num2, target_file, method)
    df = extract_scanned_pdf(
        pdf_path=target_file,
        pages=[1],  # 默认提取第一页
        dpi=800,  # 指定分辨率
        poppler_path=r"C:\Program Files\poppler-24.08.0\Library\bin"
    )

    # 保存结果为一csv文件
    df.to_csv("output.csv", index=False, encoding='utf-8-sig')
    # print("提取完成，结果已保存到 output.csv")

    # 对该csv文件依据关键词进行检索
    get_data(RAW_FILE="output.csv",OUT_FILE="reserves.csv",KEYWORD=keyword)


if __name__ == "__main__":
    cli()



