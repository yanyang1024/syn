import os
import json
import time
from typing import List, Dict, Optional, Any, Tuple

import requests
import argparse
import hashlib
import re
from glob import glob


# Baidu AppBuilder intelligent web search endpoint
WEB_SEARCH_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"


def _repo_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


# 目录常量
XUEYUAN_CACHE_DIR = os.path.join(_repo_dir(), "xueyuan_info", "baidu_cache")
XUEYUAN_QWEN_DIR = os.path.join(_repo_dir(), "xueyuan_info", "qwen_xy")
PERSON_CACHE_DIR = os.path.join(_repo_dir(), "person_info", "baidu_cache")
PERSON_QWEN_DIR = os.path.join(_repo_dir(), "person_info", "qwen_people")


def _ensure_dirs():
    for d in [
        os.path.join(_repo_dir(), "xueyuan_info"),
        os.path.join(_repo_dir(), "person_info"),
        XUEYUAN_CACHE_DIR,
        XUEYUAN_QWEN_DIR,
        PERSON_CACHE_DIR,
        PERSON_QWEN_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def _slugify(text: str) -> str:
    # 保留中文、字母、数字和下划线，其余替换为下划线
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    return text.strip("_")[:80]


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===================== Qwen (DashScope) =====================
def call_qwen_json(
    prompt: str,
    DASHSCOPE_API_KEY: Optional[str] = None,
    model: str = "qwen-plus",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    timeout: int = 30,
) -> Any:
    """
    Call Tongyi Qianwen via DashScope and parse JSON output.
    - Forces JSON via response_format=json_object.
    - Retries up to 5 times.
    Returns dict or list (parsed JSON), or {} on persistent failure.
    """
    if not DASHSCOPE_API_KEY:
        DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("Missing DASHSCOPE_API_KEY.")

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable",
    }
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
    }

    for attempt in range(1, 6):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content: str = data["output"]["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception as exc:
            print(f"[call_qwen_json] attempt {attempt}/5 failed: {exc}")
        if attempt < 5:
            time.sleep(2 ** attempt * 0.5)
    return {}


# ===================== Baidu smart search =====================
def baidu_smart_search(
    api_key: str,
    query: str,
    top_k: int = 10,
    recency: str = "year",
    site_filter: Optional[List[str]] = None,
    timeout: int = 10,
) -> List[Dict]:
    """
    Call Baidu AppBuilder intelligent web search and return normalized list of results.
    Returns list of dicts: {title, url, content, date, type}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role": "user", "content": query}],
        "search_source": "baidu_search_v2",
        "resource_type_filter": [{"type": "web", "top_k": min(top_k, 20)}],
        "search_recency_filter": recency,
    }
    if site_filter:
        payload["search_filter"] = {"match": {"site": site_filter}}

    resp = requests.post(WEB_SEARCH_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Baidu smart search failed, HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    refs = data.get("references", [])
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "date": r.get("date", ""),
            "type": r.get("type", "web"),
        }
        for r in refs
    ]
    return results


# ===================== Local data =====================
def get_school_name() -> List[str]:
    path = os.path.join(_repo_dir(), "sc_l.txt")
    schools: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                schools.append(s)
    return schools


def get_major_name() -> List[str]:
    path = os.path.join(_repo_dir(), "zy_l.txt")
    majors: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                majors.append(s)
    return majors


# ===================== Stage 1: prompts for colleges =====================
def get_xueyuan_prompt(sc_list: List[str], mj_list: List[str]) -> Dict[str, List[str]]:
    prompt_map: Dict[str, List[str]] = {}
    for sc in sc_list:
        variants = []
        for mj in mj_list:
            variants.extend([
                f"{sc}{mj} 专业 所在 学院 名称",
                f"{sc} {mj} 专业 属于 哪个 学院",
                f"{sc} {mj} 学院 是 什么",
            ])
        prompt_map[sc] = variants
    return prompt_map


def get_xueyuan_pre_info(prompt_map: Dict[str, List[str]], api_key: str, resume: bool = True) -> Dict[str, List[Dict]]:
    """
    分步+断点：按学校逐条 prompt 检索并缓存到 xueyuan_info/baidu_cache。
    多次运行会自动复用已存在的缓存，失败即跳过继续。
    返回每个学校聚合后的去重结果。
    """
    _ensure_dirs()

    out: Dict[str, List[Dict]] = {sc: [] for sc in prompt_map.keys()}
    for sc, prompts in prompt_map.items():
        school_slug = _slugify(sc)
        # 单个 prompt 缓存
        for p in prompts:
            fname = f"{school_slug}__{_hash(p)}.json"
            fpath = os.path.join(XUEYUAN_CACHE_DIR, fname)
            if resume and os.path.exists(fpath):
                # 已缓存
                continue
            try:
                res = baidu_smart_search(api_key, p, top_k=20, recency="month")
            except Exception as e:
                print(f"[学院检索] {sc} | prompt 失败: {e}")
                res = []
            _save_json(fpath, {"school": sc, "prompt": p, "results": res})
            time.sleep(0.25)

        # 聚合该学校所有缓存
        merged: List[Dict] = []
        for fp in glob(os.path.join(XUEYUAN_CACHE_DIR, f"{school_slug}__*.json")):
            try:
                data = _load_json(fp)
                merged.extend(data.get("results", []))
            except Exception:
                pass
        seen = set()
        dedup: List[Dict] = []
        for r in merged:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                dedup.append(r)
        out[sc] = dedup
        _save_json(os.path.join(_repo_dir(), "xueyuan_info", f"baidu_xy_{school_slug}.json"),
                   {"school": sc, "results": dedup})

    _save_json(os.path.join(_repo_dir(), "xueyuan_info", "xueyuan_pre_info.json"), out)
    return out


def get_xy(xueyuan_pre_info: Dict[str, List[Dict]], das_key: Optional[str], resume: bool = True) -> Dict[str, List[Dict]]:
    """
    Extract colleges via LLM. Input: {school: [web_result,...]}
    Output: {school: [{"school":..., "xueyuan":...}, ...]}
    """
    _ensure_dirs()

    out: Dict[str, List[Dict]] = {}
    for sc, items in xueyuan_pre_info.items():
        # 如已缓存，则直接复用
        out_path = os.path.join(XUEYUAN_QWEN_DIR, f"xy_{_slugify(sc)}.json")
        if resume and os.path.exists(out_path):
            try:
                cached = _load_json(out_path)
                out[sc] = cached.get("items", [])
                continue
            except Exception:
                pass
        chunks = []
        for it in items[:50]:
            title = it.get("title", "")
            content = it.get("content", "")
            url = it.get("url", "")
            if title or content:
                chunks.append(f"Title: {title}\nURL: {url}\nExcerpt: {content}")
        context = "\n\n".join(chunks)
        prompt = (
            "请阅读以下搜索结果片段，从中提取所有涉及到的学院名称。\n"
            "请返回一个JSON对象，键为 'items'，其值为一个数组；数组中每个元素为："
            " {\"school\": \"学校名称或 'unknow'\", \"xueyuan\": \"学院名称\"}。\n\n"
            f"```\n{context}\n```\n"
        )
        try:
            llm_res = call_qwen_json(prompt, DASHSCOPE_API_KEY=das_key)
        except Exception as e:
            print(f"[get_xy] LLM failed: {e}")
            llm_res = {}

        items_out: List[Dict] = []
        if isinstance(llm_res, dict):
            items_out = llm_res.get("items", []) or llm_res.get("data", []) or []
        elif isinstance(llm_res, list):
            items_out = llm_res

        norm: List[Dict] = []
        seen_xy = set()
        for x in items_out:
            xy = (x or {}).get("xueyuan") or (x or {}).get("学院")
            if not xy or xy in seen_xy:
                continue
            seen_xy.add(xy)
            norm.append({"school": sc, "xueyuan": xy})
        out[sc] = norm

        _save_json(out_path, {"school": sc, "items": norm})
        time.sleep(0.2)

    _save_json(os.path.join(_repo_dir(), "xueyuan_info", "qwen_result_xueyuan_all.json"), out)
    return out


# ===================== Stage 2: search dean/vice-dean =====================
def get_person_prompt(sc_xy: Dict[str, List[Dict]], sc_list: List[str], key: str, resume: bool = True) -> Tuple[List[Dict], List[Dict]]:
    _ensure_dirs()

    dean_results: List[Dict] = []
    vdean_results: List[Dict] = []
    idx = 0
    for school in sc_list:
        xy_list = [x.get("xueyuan") for x in sc_xy.get(school, []) if x.get("xueyuan")]
        for xy in xy_list:
            q1 = f"{school}{xy} 院长"
            q2 = f"{school}{xy} 副院长"
            # dean cache
            base = f"{_slugify(school)}__{_slugify(xy)}"
            f1 = os.path.join(PERSON_CACHE_DIR, f"{base}__dean__{_hash(q1)}.json")
            f2 = os.path.join(PERSON_CACHE_DIR, f"{base}__vdean__{_hash(q2)}.json")
            if resume and os.path.exists(f1):
                r1 = _load_json(f1).get("results", [])
            else:
                try:
                    r1 = baidu_smart_search(key, q1, top_k=20, recency="year")
                except Exception as e:
                    print(f"[院长检索] {school}-{xy} 失败: {e}")
                    r1 = []
                _save_json(f1, {"school": school, "xueyuan": xy, "query": q1, "results": r1})
                time.sleep(0.2)
            if resume and os.path.exists(f2):
                r2 = _load_json(f2).get("results", [])
            else:
                try:
                    r2 = baidu_smart_search(key, q2, top_k=20, recency="year")
                except Exception as e:
                    print(f"[副院长检索] {school}-{xy} 失败: {e}")
                    r2 = []
                _save_json(f2, {"school": school, "xueyuan": xy, "query": q2, "results": r2})
                time.sleep(0.2)
            dean_results.append({"school": school, "xueyuan": xy, "query": q1, "results": r1})
            vdean_results.append({"school": school, "xueyuan": xy, "query": q2, "results": r2})
            _save_json(os.path.join(_repo_dir(), "person_info", f"baidu_dean_{idx}.json"), dean_results[-1])
            _save_json(os.path.join(_repo_dir(), "person_info", f"baidu_vdean_{idx}.json"), vdean_results[-1])
            idx += 1
            time.sleep(0.1)

    _save_json(os.path.join(_repo_dir(), "person_info", "baidu_dean_all.json"), dean_results)
    _save_json(os.path.join(_repo_dir(), "person_info", "baidu_vdean_all.json"), vdean_results)
    return dean_results, vdean_results


def get_person_prompt_llm(entries: List[Dict], tag: str, das_key: Optional[str], resume: bool = True) -> List[Dict]:
    """
    Extract people (dean/vice-dean) from web results via LLM.
    Returns list of dicts: {school, xuyyuan, name, zhiwu}
    """
    save_dir = os.path.join(_repo_dir(), "person_info")
    os.makedirs(save_dir, exist_ok=True)

    results: List[Dict] = []
    for e in entries:
        school = e.get("school", "")
        xy = e.get("xueyuan", "")
        web_items = e.get("results", [])
        chunks = []
        for it in web_items[:40]:
            title = it.get("title", "")
            content = it.get("content", "")
            url = it.get("url", "")
            if title or content:
                chunks.append(f"Title: {title}\nURL: {url}\nExcerpt: {content}")
        context = "\n\n".join(chunks)
        prompt = (
            f"请从以下网页片段中抽取{school}{xy}{tag}的人名。\n"
            "请返回一个JSON对象，键为 'items'，其值为一个数组；数组中每个元素为："
            " {\"school\": \"学校名称\", \"xuyyuan\": \"学院名称\", \"name\": \"人名\", \"zhiwu\": \"院长\" 或 \"副院长\"}。\n\n"
            f"```\n{context}\n```\n"
        )
        # 缓存文件（每 school+xy+tag 一个）
        out_fp = os.path.join(PERSON_QWEN_DIR, f"{_slugify(school)}__{_slugify(xy)}__{_slugify(tag)}.json")
        if resume and os.path.exists(out_fp):
            try:
                cached = _load_json(out_fp)
                items = cached.get("items", [])
            except Exception:
                items = []
        else:
            try:
                llm_res = call_qwen_json(prompt, DASHSCOPE_API_KEY=das_key)
            except Exception as ex:
                print(f"[人员抽取] LLM 失败: {ex}")
                llm_res = {}

            items = []
            if isinstance(llm_res, dict):
                items = llm_res.get("items", []) or llm_res.get("data", []) or []
            elif isinstance(llm_res, list):
                items = llm_res
            _save_json(out_fp, {"school": school, "xuyyuan": xy, "tag": tag, "items": items})

        for it in items:
            name = (it or {}).get("name") or (it or {}).get("person")
            if not name:
                continue
            results.append({
                "school": school,
                "xuyyuan": xy,
                "name": name,
                "zhiwu": "院长" if tag == "院长" else "副院长",
            })

    # dedupe
    seen = set()
    deduped: List[Dict] = []
    for r in results:
        k = (r["school"], r["xuyyuan"], r["name"], r["zhiwu"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)

    with open(os.path.join(save_dir, f"qwen_result_person_{'dean' if tag=='院长' else 'vdean'}.json"), "w", encoding="utf-8") as f:
        json.dump({"items": deduped}, f, ensure_ascii=False, indent=2)
    return deduped


def get_person(dean_entries: List[Dict], vdean_entries: List[Dict], das_key: Optional[str], resume: bool = True) -> List[Dict]:
    dean_list = get_person_prompt_llm(dean_entries, "院长", das_key, resume=resume)
    vdean_list = get_person_prompt_llm(vdean_entries, "副院长", das_key, resume=resume)
    final = dean_list + vdean_list
    save_path = os.path.join(_repo_dir(), "person_info", "final_result.json")
    _save_json(save_path, {"items": final})
    return final


# ===================== Main =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学院与院长信息抓取与抽取（支持断点重启）")
    parser.add_argument("--stage", choices=[
        "search-xueyuan", "extract-xueyuan", "search-person", "extract-person", "all"
    ], default="all", help="执行阶段")
    parser.add_argument("--schools", type=str, default="", help="仅处理指定学校，逗号分隔")
    parser.add_argument("--resume", action="store_true", default=True, help="启用断点续跑（默认开启）")
    parser.add_argument("--no-resume", action="store_true", default=False, help="关闭断点续跑（强制重新请求）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理前N所学校")
    args = parser.parse_args()

    _ensure_dirs()

    baidu_key = os.getenv("APPBUILDER_API_KEY", "").strip()
    das_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not baidu_key:
        print("[警告] 未设置 APPBUILDER_API_KEY，相关阶段将被跳过。")
    if not das_key:
        print("[警告] 未设置 DASHSCOPE_API_KEY，相关阶段将被跳过。")

    # 读取学校/专业
    sc_list = get_school_name()
    mj_list = get_major_name()
    if args.schools:
        wanted = set([s.strip() for s in args.schools.split(",") if s.strip()])
        sc_list = [s for s in sc_list if s in wanted]
    if args.limit and args.limit > 0:
        sc_list = sc_list[: args.limit]
    resume_flag = False if args.no_resume else True
    print(f"将处理 {len(sc_list)} 所学校，{len(mj_list)} 个专业。阶段: {args.stage}，断点续跑: {resume_flag}")

    # 阶段 1：院系检索
    if args.stage in ("search-xueyuan", "all") and baidu_key:
        prompts = get_xueyuan_prompt(sc_list, mj_list)
        get_xueyuan_pre_info(prompts, baidu_key, resume=resume_flag)

    # 阶段 2：院系抽取
    if args.stage in ("extract-xueyuan", "all") and das_key:
        # 从聚合文件加载（若存在）
        pre_info: Dict[str, List[Dict]] = {sc: [] for sc in sc_list}
        for sc in sc_list:
            fp = os.path.join(_repo_dir(), "xueyuan_info", f"baidu_xy_{_slugify(sc)}.json")
            if os.path.exists(fp):
                try:
                    data = _load_json(fp)
                    pre_info[sc] = data.get("results", [])
                except Exception:
                    pass
        get_xy(pre_info, das_key, resume=resume_flag)

    # 阶段 3：人员检索（院长/副院长）
    if args.stage in ("search-person", "all") and baidu_key:
        # 从院系抽取结果加载
        sc_xy_map: Dict[str, List[Dict]] = {sc: [] for sc in sc_list}
        for sc in sc_list:
            fp = os.path.join(XUEYUAN_QWEN_DIR, f"xy_{_slugify(sc)}.json")
            if os.path.exists(fp):
                try:
                    data = _load_json(fp)
                    sc_xy_map[sc] = data.get("items", [])
                except Exception:
                    pass
        get_person_prompt(sc_xy_map, sc_list, baidu_key, resume=resume_flag)

    # 阶段 4：人员抽取
    if args.stage in ("extract-person", "all") and das_key:
        # 优先从聚合文件加载 entries
        dean_entries: List[Dict] = []
        vdean_entries: List[Dict] = []
        dean_fp = os.path.join(_repo_dir(), "person_info", "baidu_dean_all.json")
        vdean_fp = os.path.join(_repo_dir(), "person_info", "baidu_vdean_all.json")
        if os.path.exists(dean_fp):
            try:
                dean_entries = _load_json(dean_fp)
            except Exception:
                pass
        if os.path.exists(vdean_fp):
            try:
                vdean_entries = _load_json(vdean_fp)
            except Exception:
                pass
        if not dean_entries or not vdean_entries:
            # 回退：从缓存拼装
            for fp in glob(os.path.join(PERSON_CACHE_DIR, "*__dean__*.json")):
                try:
                    data = _load_json(fp)
                    dean_entries.append(data)
                except Exception:
                    pass
            for fp in glob(os.path.join(PERSON_CACHE_DIR, "*__vdean__*.json")):
                try:
                    data = _load_json(fp)
                    vdean_entries.append(data)
                except Exception:
                    pass
        items = get_person(dean_entries, vdean_entries, das_key, resume=resume_flag)
        print(f"完成，共抽取 {len(items)} 条人员记录 -> person_info/final_result.json")
