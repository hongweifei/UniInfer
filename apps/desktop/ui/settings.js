(() => {
  window.mountSettings = ({ info, authHeaders }) => {
    const dialog = document.getElementById("settings-dialog");
    const open = document.getElementById("settings-button");
    const close = document.getElementById("settings-close");
    const path = document.getElementById("config-path");
    const editor = document.getElementById("config-json");
    const status = document.getElementById("settings-status");
    const load = document.getElementById("config-load");
    const save = document.getElementById("config-save");
    const modelPath = document.getElementById("model-path");
    const inspect = document.getElementById("model-inspect");
    const importModel = document.getElementById("model-import");
    const result = document.getElementById("model-result");

    const request = async (url, options = {}) => {
      const response = await fetch(`${info.base}${url}`, {
        ...options,
        headers: { ...authHeaders(), ...(options.headers ?? {}) },
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(body.message ?? `HTTP ${response.status}`);
      }
      return body;
    };

    const loadConfig = async () => {
      status.textContent = "读取中…";
      try {
        const body = await request("/v1/config");
        path.textContent = body.path ?? "未绑定配置文件";
        editor.value = JSON.stringify(body.config, null, 2);
        status.textContent = body.token_configured
          ? "token 已配置；<redacted> 回写会保留原值"
          : "token 未配置";
      } catch (error) {
        status.textContent = error.message ?? String(error);
      }
    };

    open.addEventListener("click", () => {
      dialog.showModal();
      void loadConfig();
    });
    close.addEventListener("click", () => dialog.close());
    load.addEventListener("click", () => void loadConfig());
    save.addEventListener("click", async () => {
      status.textContent = "保存中…";
      try {
        const config = JSON.parse(editor.value);
        const body = await request("/v1/config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config),
        });
        status.textContent = body.restart_required
          ? "已保存；重启后生效"
          : "已保存";
      } catch (error) {
        status.textContent = error.message ?? String(error);
      }
    });

    inspect.addEventListener("click", async () => {
      result.textContent = "检查中…";
      importModel.disabled = true;
      try {
        const body = await request("/v1/models/inspect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: modelPath.value.trim() }),
        });
        result.textContent = JSON.stringify(body, null, 2);
        importModel.disabled = false;
      } catch (error) {
        result.textContent = error.message ?? String(error);
      }
    });

    importModel.addEventListener("click", async () => {
      result.textContent = "导入中…";
      try {
        const body = await request("/v1/models/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: modelPath.value.trim() }),
        });
        result.textContent = JSON.stringify(body, null, 2);
        await loadConfig();
        status.textContent = "模型已导入；重启后加载";
      } catch (error) {
        result.textContent = error.message ?? String(error);
      }
    });
  };
})();
