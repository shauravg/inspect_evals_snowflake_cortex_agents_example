"""Quick script to print raw SSE events from the Cortex Agents :run endpoint.

Usage:
    conda run -n inspect_evals python -m cortex_inspect.debug_sse \
        --db mydb --schema myschema --agent my_agent \
        --message "What is 2+2?"
"""
import argparse
import asyncio
import os

import aiohttp


async def stream_raw(db: str, schema: str, agent: str, message: str) -> None:
    token = os.environ.get("SNOWFLAKE_CORTEX_TOKEN", "")
    base_url = os.environ.get("SNOWFLAKE_CORTEX_BASE_URL", "").rstrip("/")

    url = f"{base_url}/api/v2/databases/{db}/schemas/{schema}/agents/{agent}:run"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": message}]}]}

    print(f"POST {url}\n")
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, headers=headers) as resp:
            print(f"HTTP {resp.status}  content-type: {resp.headers.get('content-type')}\n")
            print("─" * 60)
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8")
                print(repr(line))
            print("─" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--message", default="What is 2+2?")
    args = parser.parse_args()
    asyncio.run(stream_raw(args.db, args.schema, args.agent, args.message))


if __name__ == "__main__":
    main()
