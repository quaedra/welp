"""Calibration text: wikitext-2 train + Python stdlib + llama.cpp C/C++ sources and docs.
Excludes every file used by eval (wiki.test, data/code.txt sources). Writes data/calib_{wiki,code}.txt."""
import glob, os, random
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_files = {"asyncio/base_events.py", "json/decoder.py", "argparse.py", "dataclasses.py", "functools.py",
              "pathlib/_local.py", "http/client.py", "email/_header_value_parser.py", "textwrap.py", "heapq.py"}
random.seed(0)
py = [p for p in glob.glob("/usr/lib/python3.14/**/*.py", recursive=True)
      if os.path.relpath(p, "/usr/lib/python3.14") not in eval_files and "site-packages" not in p and "/test" not in p]
cc = glob.glob(f"{root}/llama.cpp/src/*.cpp") + glob.glob(f"{root}/llama.cpp/ggml/src/ggml-cuda/*.cu") + glob.glob(f"{root}/llama.cpp/tools/server/*.cpp")
md = glob.glob(f"{root}/llama.cpp/docs/**/*.md", recursive=True)
files = py + cc + md
random.shuffle(files)
with open(f"{root}/data/calib_code.txt", "w") as out:
    for p in files:
        try: txt = open(p, encoding="utf-8").read()
        except Exception: continue
        out.write(f"# file: {os.path.basename(p)}\n{txt}\n")
os.system(f"cp {root}/data/wikitext-2-raw/wiki.train.raw {root}/data/calib_wiki.txt")
print(len(py), len(cc), len(md), os.path.getsize(f"{root}/data/calib_code.txt") >> 20, "MB code")
