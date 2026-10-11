from flask import Flask, render_template, request, jsonify
from src.agent import ask, clear_memory

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()
        conversation_id = str(data.get("conversation_id", "default")).strip() or "default"

        if not question:
            return jsonify({
                "success": False,
                "error": "Please enter a question."
            }), 400

        result = ask(question, conversation_id=conversation_id)

        return jsonify({
            "success": True,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "tools_used": result.get("tools_used", [])
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


@app.route("/clear", methods=["POST"])
def clear_chat():
    try:
        data = request.get_json(silent=True) or {}
        conversation_id = str(data.get("conversation_id", "default")).strip() or "default"
        clear_memory(conversation_id=conversation_id)

        return jsonify({
            "success": True,
            "message": "Conversation cleared."
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Travel Policy Assistant"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)