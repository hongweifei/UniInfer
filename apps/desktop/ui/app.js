const { invoke } = window.__TAURI__.core;
      const messages = document.getElementById("messages");
      const status = document.getElementById("status");
      const modelSelect = document.getElementById("model");
      const form = document.getElementById("composer");
      const input = document.getElementById("input");
      const sendButton = document.getElementById("send");
      const stopButton = document.getElementById("stop");

      let info = null;
      let model = null;
      let controller = null;

      function authHeaders() {
        return { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" };
      }

      function addTurn(role, text = "") {
        const node = document.createElement("div");
        node.className = `turn ${role}`;
        node.textContent = text;
        messages.appendChild(node);
        messages.scrollTop = messages.scrollHeight;
        return node;
      }

      function setBusy(busy) {
        sendButton.disabled = busy;
        stopButton.disabled = !busy;
        status.textContent = busy ? "生成中…" : `${model ?? "未加载模型"}`;
      }

      async function boot() {
        const desktop = await invoke("api_info");
        if (!desktop.info) {
          status.textContent = "模型加载失败";
          sendButton.disabled = true;
          const turn = addTurn(
            "error",
            `无法加载模型：${desktop.error ?? "unknown error"}\n清单：${desktop.manifest}\n` +
              "请检查清单路径，或设置 UNIINFER_MANIFEST；固定件可用 models/fetch.sh 生成。"
          );
          turn.classList.add("error");
          return;
        }
        info = desktop.info;
        const response = await fetch(`${info.base}/v1/models`, { headers: authHeaders() });
        if (!response.ok) throw new Error(`models: HTTP ${response.status}`);
        const body = await response.json();
        modelSelect.replaceChildren(
          ...body.data.map((entry) => {
            const option = document.createElement("option");
            option.value = entry.id;
            option.textContent = entry.id;
            return option;
          })
        );
        model = body.data[0]?.id ?? null;
        modelSelect.value = model ?? "";
        modelSelect.disabled = body.data.length < 2;
        modelSelect.addEventListener("change", () => {
          model = modelSelect.value;
          status.textContent = model;
        });
        setBusy(false);
        status.textContent = `${model} · ${desktop.manifest}`;
        input.focus();
      }

      function parseFrame(frame) {
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6);
          if (payload === "[DONE]") continue;
          try {
            const event = JSON.parse(payload);
            if (event.error) throw new Error(event.error.message ?? "server error");
            const delta = event.choices?.[0]?.delta?.content;
            if (delta) return delta;
          } catch (error) {
            if (error instanceof SyntaxError) continue;
            throw error;
          }
        }
        return null;
      }

      async function send(text) {
        const userTurn = addTurn("user", text);
        const assistantTurn = addTurn("assistant", "");
        controller = new AbortController();
        setBusy(true);
        let buffer = "";
        try {
          const response = await fetch(`${info.base}/v1/chat/completions`, {
            method: "POST",
            headers: authHeaders(),
            body: JSON.stringify({
              model,
              messages: [{ role: "user", content: text }],
              stream: true,
              max_tokens: 64,
            }),
            signal: controller.signal,
          });
          if (!response.ok) {
            const body = await response.json().catch(() => ({}));
            throw new Error(body.message ?? `HTTP ${response.status}`);
          }
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let index;
            while ((index = buffer.indexOf("\n\n")) >= 0) {
              const frame = buffer.slice(0, index);
              buffer = buffer.slice(index + 2);
              const delta = parseFrame(frame);
              if (delta) {
                assistantTurn.textContent += delta;
                messages.scrollTop = messages.scrollHeight;
              }
            }
          }
        } catch (error) {
          if (error.name === "AbortError") {
            assistantTurn.textContent += "\n[已停止]";
          } else {
            assistantTurn.classList.add("error");
            assistantTurn.textContent = error.message ?? String(error);
          }
        } finally {
          controller = null;
          setBusy(false);
          input.focus();
        }
        return userTurn;
      }

      form.addEventListener("submit", (event) => {
        event.preventDefault();
        const text = input.value.trim();
        if (!text || controller) return;
        input.value = "";
        void send(text);
      });

      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          form.requestSubmit();
        }
      });

      stopButton.addEventListener("click", () => controller?.abort());

      boot().catch((error) => {
        status.textContent = "启动失败";
        const turn = addTurn("error", error.message ?? String(error));
        turn.classList.add("error");
      });
