import asyncio
import os

from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.deep_research import deep_research, write_final_answer, write_final_report

app = Flask(__name__)
CORS(app)

PORT = int(os.environ.get("PORT", "3051"))


def _log(*args, **kwargs):
    print(*args, **kwargs)


# ── Research endpoint ─────────────────────────────────────────────────────

@app.route("/api/research", methods=["POST"])
def research():
    try:
        body = request.get_json(silent=True) or {}
        query = body.get("query")
        depth = body.get("depth", 3)
        breadth = body.get("breadth", 3)

        if not query:
            return jsonify({"error": "Query is required"}), 400

        _log("\nStarting research...\n")

        result = asyncio.run(deep_research(
            query=query,
            breadth=breadth,
            depth=depth,
        ))

        _log(f"\n\nLearnings:\n\n{'\n'.join(result.learnings)}")
        _log(
            f"\n\nVisited URLs ({len(result.visited_urls)}):\n\n"
            + "\n".join(result.visited_urls)
        )

        answer = write_final_answer(
            prompt=query,
            learnings=result.learnings,
        )

        return jsonify({
            "success": True,
            "answer": answer,
            "learnings": result.learnings,
            "visitedUrls": result.visited_urls,
            "relevantImages": [img.model_dump() for img in result.relevant_images],
        })

    except Exception as e:
        print("Error in research API:", e)
        return jsonify({
            "error": "An error occurred during research",
            "message": str(e),
        }), 500


# ── Generate report endpoint ──────────────────────────────────────────────

@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    try:
        body = request.get_json(silent=True) or {}
        query = body.get("query")
        depth = body.get("depth", 3)
        breadth = body.get("breadth", 3)

        if not query:
            return jsonify({"error": "Query is required"}), 400

        _log("\nStarting research...\n")

        result = asyncio.run(deep_research(
            query=query,
            breadth=breadth,
            depth=depth,
        ))

        _log(f"\n\nLearnings:\n\n{'\n'.join(result.learnings)}")
        _log(
            f"\n\nVisited URLs ({len(result.visited_urls)}):\n\n"
            + "\n".join(result.visited_urls)
        )

        report = write_final_report(
            prompt=query,
            learnings=result.learnings,
            visited_urls=result.visited_urls,
            relevant_images=result.relevant_images,
        )

        return report

    except Exception as e:
        print("Error in generate report API:", e)
        return jsonify({
            "error": "An error occurred during research",
            "message": str(e),
        }), 500


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Deep Research API running on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
