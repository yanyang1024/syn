# Text_Analyzer.py
from mcp.server.fastmcp import FastMCP
from collections import Counter
import re

# 创建一个 MCP_server
mcp = FastMCP("TextAnalyzer")

@mcp.prompt()
def Textanalyzer_prompt(text: str) -> str:
    return f"你是一个文本分析助手，请分析以下文本：{text}"

# 一个分析文本基本信息的工具
@mcp.tool()
def analyze_text(text: str) -> str:
    """ 分析文本的基本信息,
    如单词数，句子数,
    参数 text 是要分析的文本    
    """
    words = text.split()
    sentences = len(re.split(r'[.!?]+',text)) - 1
    word_counts = Counter(words)
    common_words = word_counts.most_common(5)
    return f""" 文本分析结果：
1.单词数：{len(words)}
2.句子数：{sentences}
3.最常见的单词：{'，'.join([f"{word}({count})" for word, count in common_words])}
    """

# 一个关键词提取工具
@mcp.tool()
def find_keywords(text: str,num_keywords: int = 5) ->str:
    """ 从文本中剔除常用停用词后基于词频提取关键词,
    参数 text 为要分析的文本,
    num_keywords 为要提取的单词数量
    """
    delete_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "but","that","is","are","it","have","has"}
    words = [word.lower() for word in re.findall(r'\b\w+\b',text) if word not in delete_words]
    word_counts = Counter(words)
    keywords = word_counts.most_common(num_keywords)
    return f"关键词：{'，'.join([f"{word}({count})" for word, count in keywords])}"
if __name__  == "__main__":
    mcp.run(transport="stdio")