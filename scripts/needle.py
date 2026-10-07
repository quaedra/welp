"""Hide a passcode at several depths of a ~214k-token code prompt; ask the server to return it."""
import json, sys, urllib.request
port = sys.argv[1]
code = open("data/calib_code.txt", encoding="utf-8").read()[:685000]
ok = 0
for frac, key in ((0.1, "731-PEAR-904"), (0.5, "268-OTTER-115"), (0.9, "540-MAPLE-372")):
    i = code.rfind("\n", 0, int(len(code) * frac))
    text = code[:i] + f"\n# NOTE: the deployment passcode is {key}. Remember it.\n" + code[i:]
    body = {"temperature": 0, "max_tokens": 40, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text + "\n\nWhat is the deployment passcode mentioned in a NOTE comment above? Reply with the passcode only."}]}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
    ans = r["choices"][0]["message"]["content"].strip()
    ok += key in ans
    print(f"depth {frac:.0%}: {'OK ' if key in ans else 'MISS'} -> {ans[:60]!r}", flush=True)
print(f"{ok}/3")
