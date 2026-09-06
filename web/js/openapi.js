// ComfyUI-OpenAPI 前端扩展：在 OpenAPIImageGenerator 节点上提供“获取模型列表”按钮
// 注意：必须用绝对路径导入。本文件位于 web/js/（两层深），服务路径为
// /extensions/<pkg>/js/openapi.js —— 相对路径 "../../scripts/app.js" 会错误解析到
// /extensions/scripts/app.js 而 404（实机 QA 已验证）。绝对路径与嵌套深度无关。
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_CLASS = "OpenAPIImageGenerator";
const ROUTE_FETCH = "/comfyui_openapi/fetch_models";
const ROUTE_CACHE = "/comfyui_openapi/cache";

// 统一 toast 入口；extensionManager 缺失时降级为 console.warn
function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) {
    t.add({ severity, summary, detail, life: 3000 });
  } else {
    console.warn(`[OpenAPI] ${severity}: ${summary}${detail ? " - " + detail : ""}`);
  }
}

// 按钮回调：拉取远端模型列表并回填 model 下拉选项
async function fetchModels(node, buttonWidget) {
  const find = (name) => node.widgets?.find((w) => w.name === name) ?? null;
  const baseUrl = (find("base_url")?.value ?? "").trim();
  if (!baseUrl) {
    toast("error", "请先填写 base_url");
    return;
  }
  const apiKey = find("api_key")?.value ?? "";
  // v0.2：读取 protocol COMBO（openai/dashscope）；旧工作流节点无此控件时默认 openai
  const protocol = find("protocol")?.value || "openai";
  const originalLabel = buttonWidget.label;
  buttonWidget.disabled = true;
  buttonWidget.label = "获取中…";
  try {
    const res = await api.fetchApi(ROUTE_FETCH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_url: baseUrl, api_key: apiKey, protocol: protocol }),
    });
    const json = await res.json();
    if (res.ok && json?.ok && Array.isArray(json.models)) {
      if (json.models.length > 0) {
        const modelWidget = find("model");
        if (modelWidget) {
          modelWidget.options = modelWidget.options || {};
          modelWidget.options.values = json.models;
          if (!json.models.includes(modelWidget.value)) {
            modelWidget.value = json.models[0];
          }
        }
        // 画布重绘: 新版前端 app.canvas 无 setDirtyCanvas (实机 QA 实测),
        // 以 app.graph.setDirtyCanvas 为主、canvas 为后备; 重绘失败绝不影响成功反馈
        try {
          if (app.graph && typeof app.graph.setDirtyCanvas === "function") {
            app.graph.setDirtyCanvas(true, true);
          } else if (app.canvas && typeof app.canvas.setDirtyCanvas === "function") {
            app.canvas.setDirtyCanvas(true, true);
          }
        } catch {
          // 重绘仅为视觉刷新, 失败静默忽略
        }
        toast("success", "模型列表已更新", `共 ${json.models.length} 个模型`);
      } else {
        toast("warn", "远端未返回任何模型");
      }
    } else {
      toast("error", "获取模型失败", json?.error ?? `HTTP ${res.status}`);
    }
  } catch (e) {
    toast("error", "获取模型失败", String(e?.message ?? e));
  } finally {
    buttonWidget.disabled = false;
    buttonWidget.label = originalLabel;
  }
}

app.registerExtension({
  name: "ComfyUI.OpenAPI.FetchModels",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;

    // 节点创建后追加按钮 widget（保留并先执行原始 onNodeCreated）
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const r = origOnNodeCreated?.apply(this, args);
      const w = this.addWidget("button", "获取模型列表", null, () => {
        fetchModels(this, w);
      });
      w.name = "Fetch Models"; // 供测试按 name 查找
      return r;
    };

    // 加载配置后按 base_url 从缓存预填 api_key（尽力而为，失败静默）
    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = origOnConfigure?.apply(this, arguments);
      (async () => {
        try {
          const apiKeyW = this.widgets?.find((x) => x.name === "api_key");
          const baseUrlW = this.widgets?.find((x) => x.name === "base_url");
          const baseUrl = (baseUrlW?.value ?? "").trim();
          if (apiKeyW && (apiKeyW.value == null || apiKeyW.value === "") && baseUrl) {
            const res = await api.fetchApi(`${ROUTE_CACHE}?base_url=${encodeURIComponent(baseUrl)}`);
            const json = await res.json();
            if (json?.ok && json.api_key) apiKeyW.value = json.api_key;
          }
        } catch {
          /* 预填失败可忽略 */
        }
      })();
      return r;
    };
  },
});
