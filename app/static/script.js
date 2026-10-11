const input = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const chatBody = document.getElementById("chatBody");
const thinking = document.getElementById("thinking");
const welcomeBlock = document.getElementById("welcomeBlock");
const clearBtn = document.getElementById("clearBtn");
const pageTitle = document.getElementById("pageTitle");

const conversationId = "dashboard-default";

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function scrollChat() {
  chatBody.scrollTop = chatBody.scrollHeight;
}

function addMessage(role, text, result = {}) {
  if (welcomeBlock) welcomeBlock.style.display = "none";

  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `message-avatar ${role === "user" ? "user" : "ai"}`;
  avatar.textContent = role === "user" ? "YOU" : "TA";

  const inner = document.createElement("div");
  inner.className = "message-inner";

  const textEl = document.createElement("div");
  textEl.className = "message-text";
  textEl.innerHTML = escapeHtml(text);

  inner.appendChild(textEl);

  if (role === "assistant") {
    if (result.sources?.length) {
      const sourceWrap = document.createElement("div");
      sourceWrap.className = "source-tags";
      result.sources.forEach(s => {
        const tag = document.createElement("span");
        tag.className = "source-tag";
        tag.textContent = `Policy · ${s.source || "source"}`;
        sourceWrap.appendChild(tag);
      });
      inner.appendChild(sourceWrap);
    }

    if (result.tools_used?.length) {
      const toolWrap = document.createElement("div");
      toolWrap.className = "tool-tags-inline";
      result.tools_used.forEach(t => {
        const tag = document.createElement("span");
        tag.className = "tool-tag";
        tag.textContent = `Tool · ${t}`;
        toolWrap.appendChild(tag);
      });
      inner.appendChild(toolWrap);
    }
  }


  wrapper.appendChild(avatar);
  wrapper.appendChild(inner);
  chatBody.appendChild(wrapper);
  scrollChat();
}

async function askQuestion(question) {
  const q = String(question || "").trim();
  if (!q || sendBtn.disabled) return;

  addMessage("user", q);
  input.value = "";
  autoResize();

  thinking.classList.add("active");
  sendBtn.disabled = true;

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, conversation_id: conversationId })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Unable to process the request.");
    }

    addMessage("assistant", data.answer || "No answer returned.", data);
  } catch (error) {
    addMessage("assistant", `I couldn't complete that request.\n\n${error.message}`);
  } finally {
    thinking.classList.remove("active");
    sendBtn.disabled = false;
    input.focus();
  }
}

async function clearConversation() {
  clearBtn.disabled = true;
  try {
    const response = await fetch("/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversationId })
    });

    const data = await response.json();
    if (!response.ok || !data.success) throw new Error(data.error || "Unable to clear conversation.");

    chatBody.innerHTML = "";
    chatBody.appendChild(welcomeBlock);
    welcomeBlock.style.display = "flex";
  } catch (error) {
    alert(error.message);
  } finally {
    clearBtn.disabled = false;
  }
}

function autoResize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 100) + "px";
}

sendBtn.addEventListener("click", () => askQuestion(input.value));

input.addEventListener("input", autoResize);
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askQuestion(input.value);
  }
});

document.querySelectorAll(".quick-card").forEach(card => {
  card.addEventListener("click", () => askQuestion(card.dataset.question));
});

clearBtn.addEventListener("click", clearConversation);

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    item.classList.add("active");

    const view = item.dataset.view;
    const titles = {
      assistant: "AI Travel Assistant",
      policies: "Policy Knowledge Base",
      eligibility: "Employee Eligibility",
      validation: "Trip Validation",
      reimbursement: "Reimbursement"
    };
    pageTitle.textContent = titles[view] || "AI Travel Assistant";

    if (view !== "assistant") {
      const prompts = {
        policies: "What are the available company travel policies and what does each policy cover?",
        eligibility: "How do I check whether an employee is eligible for company travel?",
        validation: "What information is required to validate an employee trip?",
        reimbursement: "How is the reimbursable amount calculated?"
      };
      if (prompts[view]) askQuestion(prompts[view]);
    }
  });
});

input.focus();
